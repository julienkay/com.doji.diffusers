using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion pipeline 
    /// </summary>
    /// <remarks>
    /// stable_diffusion/pipeline_onnx_stable_diffusion.py from huggingface/diffusers
    /// </remarks>
    public partial class StableDiffusionPipeline : DiffusionPipeline, IDisposable {

        private Ops _ops;

        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            BackendType backend)
        {
            VaeDecoder = vaeDecoder;
            Tokenizer = tokenizer;
            TextEncoder = textEncoder;
            Scheduler = scheduler;
            Unet = unet;
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        public override TensorFloat Generate(
            Input prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            Input negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, float, TensorFloat> callback = null)
        {
            Profiler.BeginSample($"{GetType().Name}.Generate");

            _prompt = prompt;
            _negativePrompt = negativePrompt;
            _height = height;
            _width = width;
            _numImagesPerPrompt = numImagesPerPrompt;
            _guidanceScale = guidanceScale;
            _eta = eta;
            _latents = latents;
            CheckInputs(seed);

            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            } else if (prompt is TextInput) {
                _batchSize = 1;
            } else if (prompt is BatchInput prompts) {
                _batchSize = prompts.Sequence.Count;
            } else {
                throw new ArgumentException($"Invalid prompt argument {nameof(prompt)}");
            }

            if (latents == null) {
                _seed = seed != null ? seed : unchecked((uint)new System.Random().Next());
            }

            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            Profiler.BeginSample("Encode Prompt(s)");
            TensorFloat promptEmbeds = EncodePrompt(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);
            Profiler.EndSample();

            // get the initial random noise unless the user supplied it
            TensorShape latentsShape = GetLatentsShape();
            if (latents == null) {
                Profiler.BeginSample("Generate Latents");
                latents = _ops.RandomNormal(latentsShape, 0, 1, _seed);
                Profiler.EndSample();
            } else if (latents.shape != latentsShape) {
                throw new ArgumentException($"Unexpected latents shape, got {latents.shape}, expected {latentsShape}");
            }

            // set timesteps
            Profiler.BeginSample($"{Scheduler.GetType().Name}.SetTimesteps");
            Scheduler.SetTimesteps(numInferenceSteps);
            Profiler.EndSample();

            if (Scheduler.InitNoiseSigma > 1.0f) {
                Profiler.BeginSample("Multiply latents with scheduler sigma");
                latents = _ops.Mul(Scheduler.InitNoiseSigma, latents);
                Profiler.EndSample();
            }

            Profiler.BeginSample($"Denoising Loop");
            int i = 0;
            foreach (float t in Scheduler) {
                // expand the latents if doing classifier free guidance
                TensorFloat latentModelInput = doClassifierFreeGuidance ? _ops.Concatenate(latents, latents, 0) : latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                Profiler.BeginSample("Prepare Timestep Tensor");
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(_batchSize), t);
                Profiler.EndSample();

                Profiler.BeginSample("Execute Unet");
                TensorFloat noisePred = Unet.ExecuteModel(latentModelInput, timestep, promptEmbeds);
                Profiler.EndSample();

                // perform guidance
                if (doClassifierFreeGuidance) {
                    Profiler.BeginSample("Extend Predicted Noise For Classifier-Free Guidance");
                    (var noisePredUncond, var noisePredText) = _ops.SplitHalf(noisePred, axis: 0);
                    var tmp = _ops.Sub(noisePredText, noisePredUncond);
                    var tmp2 = _ops.Mul(guidanceScale, tmp);
                    noisePred = _ops.Add(noisePredUncond, tmp2);
                    Profiler.EndSample();
                }

                // compute the previous noisy sample x_t -> x_t-1
                Profiler.BeginSample($"{Scheduler.GetType().Name}.Step");
                var schedulerOutput = Scheduler.Step(noisePred, t, latents, eta);
                latents = schedulerOutput.PrevSample;
                Profiler.EndSample();

                if (callback != null) {
                    Profiler.BeginSample($"{GetType()} Callback");
                    callback.Invoke(i / Scheduler.Order, t, latents);
                    Profiler.EndSample();
                }

                i++;
            }
            Profiler.EndSample();

            Profiler.BeginSample($"Scale Latents");
            TensorFloat result = _ops.Div(latents, 0.18215f);
            Profiler.EndSample();

            // batch decode
            if (_batchSize > 1) {
                throw new NotImplementedException();
            }

            Profiler.BeginSample($"VaeDecoder Decode Image");
            TensorFloat image = VaeDecoder.ExecuteModel(result);
            Profiler.EndSample();

            Profiler.EndSample();
            return image;
        }

        private TensorFloat EncodePrompt(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null)
        {

            if (promptEmbeds == null) {
                Profiler.BeginSample("CLIPTokenizer Encode Input");
                var textInputs = Tokenizer.Encode(
                    text: prompt,
                    padding: Padding.MaxLength,
                    maxLength: Tokenizer.ModelMaxLength,
                    truncation: Truncation.LongestFirst
                ) as InputEncoding;
                int[] textInputIds = textInputs.InputIds.ToArray() ?? throw new Exception("Failed to get input ids from tokenizer.");
                int[] untruncatedIds = (Tokenizer.Encode(text: prompt, padding: Padding.Longest) as InputEncoding).InputIds.ToArray();

                if (untruncatedIds.Length >= textInputIds.Length && !textInputIds.ArrayEqual(untruncatedIds)) {
                    //TODO: support decoding tokens to text to be able to eventually display to user
                    UnityEngine.Debug.LogWarning("A part of your input was truncated because CLIP can only handle sequences up to " +
                    $"{Tokenizer.ModelMaxLength} tokens.");
                }
                Profiler.EndSample();

                Profiler.BeginSample("Prepare Text ID Tensor");
                using TensorInt textIdTensor = new TensorInt(new TensorShape(_batchSize, textInputIds.Length), textInputIds);
                Profiler.EndSample();

                Profiler.BeginSample("Execute TextEncoder");
                promptEmbeds = TextEncoder.ExecuteModel(textIdTensor)[0] as TensorFloat;
                Profiler.EndSample();
            }

            promptEmbeds = _ops.Repeat(promptEmbeds, numImagesPerPrompt, axis: 0);

            // get unconditional embeddings for classifier free guidance
            bool ownsPromptEmbeds = false;
            if (doClassifierFreeGuidance && negativePromptEmbeds == null) {
                List<string> uncondTokens;
                if (negativePrompt == null) {
                    uncondTokens = Enumerable.Repeat("", _batchSize).ToList();
                } else if (prompt.GetType() != negativePrompt.GetType()) {
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

                Profiler.BeginSample("CLIPTokenizer Encode Unconditioned Input");
                int maxLength = promptEmbeds.shape[1];
                var uncondInput = Tokenizer.Encode<BatchInput>(
                    text: uncondTokens,
                    padding: Padding.MaxLength,
                    maxLength: maxLength,
                    truncation: Truncation.LongestFirst
                ) as BatchEncoding;
                int[] uncondInputIds = uncondInput.InputIds as int[] ?? throw new Exception("Failed to get unconditioned input ids.");
                Profiler.EndSample();

                Profiler.BeginSample("Prepare Unconditioned Text ID Tensor");
                using TensorInt uncondIdTensor = new TensorInt(new TensorShape(_batchSize, uncondInputIds.Length), uncondInputIds);
                Profiler.EndSample();

                ownsPromptEmbeds = true;
                promptEmbeds.TakeOwnership();
                Profiler.BeginSample("Execute TextEncoder For Unconditioned Input");
                negativePromptEmbeds = TextEncoder.ExecuteModel(uncondIdTensor)[0] as TensorFloat;
                Profiler.EndSample();
            }

            if (doClassifierFreeGuidance) {
                negativePromptEmbeds = _ops.Repeat(negativePromptEmbeds, numImagesPerPrompt, axis: 0);

                // For classifier free guidance, we need to do two forward passes.
                // Here we concatenate the unconditional and text embeddings into a single batch
                // to avoid doing two forward passes
                Profiler.BeginSample("Concat Prompt Embeds For Classifier-Fee Guidance");
                TensorFloat combinedEmbeddings = _ops.Concatenate(negativePromptEmbeds, promptEmbeds, 0);
                Profiler.EndSample();

                if (ownsPromptEmbeds) {
                    promptEmbeds.Dispose();
                }

                return combinedEmbeddings;
            }

            return promptEmbeds;
        }

        private TensorShape GetLatentsShape() {
            return new TensorShape(
                _batchSize * _numImagesPerPrompt,
                4, // unet.in_channels
                _height / 8,
                _width / 8
            );
        }

        public override void Dispose() {
            base.Dispose();
            _ops?.Dispose();
        }
    }
}
