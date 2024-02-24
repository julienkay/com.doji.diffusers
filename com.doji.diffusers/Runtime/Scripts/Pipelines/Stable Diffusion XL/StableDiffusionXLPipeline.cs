using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion XL Pipeline 
    /// </summary>
    /// <remarks>
    /// pipeline_stable_diffusion_xl.py from huggingface/optimum
    /// </remarks>
    public class StableDiffusionXLPipeline : DiffusionPipeline, IDisposable {

        public VaeDecoder VaeDecoder { get; private set; }
        public ClipTokenizer Tokenizer { get; private set; }
        public ClipTokenizer Tokenizer2 { get; private set; }
        public TextEncoder TextEncoder { get; private set; }
        public TextEncoder TextEncoder2 { get; private set; }
        public Scheduler Scheduler { get; private set; }
        public Unet Unet { get; private set; }

        private List<(ClipTokenizer Tokenizer, TextEncoder TextEncoder)> Encoders { get; set; }
        
        private float VaeScaleFactor { get; set; }

        private Ops _ops;

        private List<TensorFloat> _promptEmbedsList = new List<TensorFloat>();
        private List<TensorFloat> _negativePromptEmbedsList = new List<TensorFloat>();

        /// <summary>
        /// Initializes a new Stable Diffusion XL pipeline.
        /// </summary>
        public StableDiffusionXLPipeline(
            Model vaeDecoder,
            Model textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Model unet,
            Model textEncoder2,
            ClipTokenizer tokenizer2,
            BackendType backend = BackendType.GPUCompute)
        {
            VaeDecoder = new VaeDecoder(vaeDecoder, BackendType.GPUCompute);
            Tokenizer = tokenizer;
            Tokenizer2 = tokenizer2;
            TextEncoder = new TextEncoder(textEncoder, backend);
            TextEncoder2 = new TextEncoder(textEncoder2, backend);
            Scheduler = scheduler;
            Unet = new Unet(unet, backend);
            Encoders = Tokenizer != null && TextEncoder != null
                ? new() { (Tokenizer, TextEncoder), (Tokenizer2, TextEncoder2) }
                : new() { (Tokenizer2, TextEncoder2) };

            _ops = WorkerFactory.CreateOps(backend, null);
        }

        private Embeddings EncodePrompt(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null,
            TensorFloat pooledPromptEmbeds = null,
            TensorFloat negativePooledPromptEmbeds = null)
        {

            if (promptEmbeds == null) {
                _promptEmbedsList.Clear(); 

                foreach (var (tokenizer, textEncoder) in Encoders) {
                    Profiler.BeginSample("CLIPTokenizer Encode Input");
                    var textInputs = tokenizer.Encode(
                        text: prompt,
                        padding: Padding.MaxLength,
                        maxLength: tokenizer.ModelMaxLength,
                        truncation: Truncation.LongestFirst
                    ) as InputEncoding;
                    int[] textInputIds = textInputs.InputIds.ToArray();
                    int[] untruncatedIds = (tokenizer.Encode(text: prompt, padding: Padding.Longest) as InputEncoding).InputIds.ToArray();

                    if (untruncatedIds.Length >= textInputIds.Length && !textInputIds.ArrayEqual(untruncatedIds)) {
                        //TODO: support decoding tokens to text to be able to eventually display to user
                        UnityEngine.Debug.LogWarning("A part of your input was truncated because CLIP can only handle sequences up to " +
                        $"{tokenizer.ModelMaxLength} tokens.");
                    }
                    Profiler.EndSample();

                    Profiler.BeginSample("Prepare Text ID Tensor");
                    using TensorInt textIdTensor = new TensorInt(new TensorShape(_batchSize, textInputIds.Length), textInputIds);
                    Profiler.EndSample();

                    Profiler.BeginSample("Execute TextEncoder");
                    var _promptEmbeds = textEncoder.ExecuteModel(textIdTensor);
                    Profiler.EndSample();

                    pooledPromptEmbeds = _promptEmbeds[0] as TensorFloat;
                    promptEmbeds = _promptEmbeds[-2] as TensorFloat;

                    Profiler.BeginSample($"Process Input for {numImagesPerPrompt} images per prompt.");
                    promptEmbeds = _ops.Repeat(promptEmbeds, numImagesPerPrompt, axis: 0);
                    Profiler.EndSample();

                    _promptEmbedsList.Add(promptEmbeds);
                }
                promptEmbeds = _ops.Concatenate(_promptEmbedsList, -1);
            }

            // get unconditional embeddings for classifier free guidance
            bool zeroOutNegativePrompt = negativePrompt is null && Config.ForceZerosForEmptyPrompt;
            if (doClassifierFreeGuidance && negativePromptEmbeds is null && zeroOutNegativePrompt) {
                using var zeros = TensorFloat.Zeros(promptEmbeds.shape);
                negativePromptEmbeds = zeros;
                using var zerosP = TensorFloat.Zeros(pooledPromptEmbeds.shape);
                negativePooledPromptEmbeds = zerosP;
            } else if (doClassifierFreeGuidance && negativePromptEmbeds is null) {
                negativePrompt = negativePrompt ?? "";
                List<string> uncondTokens;
                if (prompt is not null && prompt.GetType() != negativePrompt.GetType()) {
                    throw new ArgumentException($"`negativePrompt` should be the same type as `prompt`, but got {negativePrompt.GetType()} != {prompt.GetType()}.");
                } else if (negativePrompt is SingleInput) {
                    uncondTokens = Enumerable.Repeat((negativePrompt as SingleInput).Text, _batchSize).ToList();
                } else if (_batchSize != (negativePrompt as BatchInput).Sequence.Count) {
                    throw new ArgumentException($"`negativePrompt`: {negativePrompt} has batch size {(negativePrompt as BatchInput).Sequence.Count}, " +
                        $"but `prompt`: {prompt} has batch size {_batchSize}. Please make sure that passed `negativePrompt` matches " +
                        $"the batch size of `prompt`.");
                } else {
                    uncondTokens = (negativePrompt as BatchInput).Sequence as List<string>;
                }

                _negativePromptEmbedsList.Clear();
                foreach (var (tokenizer, textEncoder) in Encoders) {
                    Profiler.BeginSample("CLIPTokenizer Encode Unconditioned Input");
                    int maxLength = promptEmbeds.shape[1];
                    var uncondInput = tokenizer.Encode<BatchInput>(
                        text: uncondTokens,
                        padding: Padding.MaxLength,
                        maxLength: maxLength,
                        truncation: Truncation.LongestFirst
                    ) as BatchEncoding;
                    int[] uncondInputIds = uncondInput.InputIds as int[];
                    Profiler.EndSample();

                    Profiler.BeginSample("Prepare Unconditioned Text ID Tensor");
                    using TensorInt uncondIdTensor = new TensorInt(new TensorShape(_batchSize, uncondInputIds.Length), uncondInputIds);
                    Profiler.EndSample();

                    Profiler.BeginSample("Execute TextEncoder For Unconditioned Input");
                    var _negativePromptEmbeds = textEncoder.ExecuteModel(uncondIdTensor);
                    Profiler.EndSample();

                    negativePooledPromptEmbeds = _negativePromptEmbeds[0] as TensorFloat;
                    negativePromptEmbeds = _negativePromptEmbeds[-2] as TensorFloat;

                    // duplicate unconditional embeddings for each generation per prompt
                    Profiler.BeginSample($"Process Unconditional Input for {numImagesPerPrompt} images per prompt.");
                    negativePromptEmbeds = _ops.Repeat(promptEmbeds, numImagesPerPrompt, axis: 0);
                    Profiler.EndSample();

                    // For classifier free guidance, we need to do two forward passes.
                    // Here we concatenate the unconditional and text embeddings into a single batch
                    // to avoid doing two forward passes
                    _negativePromptEmbedsList.Add(negativePromptEmbeds);
                }
                negativePromptEmbeds = _ops.Concatenate(_negativePromptEmbedsList, -1);
            }

            pooledPromptEmbeds = _ops.Repeat(pooledPromptEmbeds, numImagesPerPrompt, axis: 0);
            negativePooledPromptEmbeds = _ops.Repeat(negativePooledPromptEmbeds, numImagesPerPrompt, axis: 0);

            return new Embeddings() {
                PromptEmbeds = promptEmbeds,
                NegativePromptEmbeds = negativePromptEmbeds,
                PooledPromptEmbeds = pooledPromptEmbeds,
                NegativePooledPromptEmbeds = negativePooledPromptEmbeds
            };
        }

        public void Dispose() {
            TextEncoder?.Dispose();
            VaeDecoder?.Dispose();
            TextEncoder?.Dispose();
            Scheduler?.Dispose();
            Unet?.Dispose();
            _ops?.Dispose();
        }

        private struct Embeddings {
            public TensorFloat PromptEmbeds { get; set; }
            public TensorFloat NegativePromptEmbeds { get; set; }
            public TensorFloat PooledPromptEmbeds { get; set; }
            public TensorFloat NegativePooledPromptEmbeds { get; set; }
        }
    }
}