using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion pipeline 
    /// </summary>
    /// <remarks>
    /// stable_diffusion/pipeline_onnx_stable_diffusion.py from huggingface/diffusers
    /// </remarks>
    public partial class StableDiffusionPipeline : DiffusionPipeline, ITxt2ImgPipeline, IDisposable {

        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            BackendType backend) : base(vaeDecoder, textEncoder, tokenizer, scheduler, unet, backend) { }

        public override Parameters GetDefaultParameters() {
            return new Parameters() {
                Height             = 512,
                Width              = 512,
                NumInferenceSteps  = 50,
                GuidanceScale      = 7.5f,
                NegativePrompt     = null,
                NumImagesPerPrompt = 1,
                Eta                = 0.0f,
                Seed               = null,
                Latents            = null,
                GuidanceRescale    = 0.0f,
                Callback           = null
            };
        }

        public override Tensor<float> Generate(Parameters parameters) {
            Profiler.BeginSample($"{GetType().Name}.Generate");

            InitGenerate(parameters);

            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            } else if (prompt is TextInput) {
                batchSize = 1;
            } else if (prompt is BatchInput prompts) {
                batchSize = prompts.Sequence.Count;
            } else {
                throw new ArgumentException($"Invalid prompt argument {nameof(prompt)}");
            }

            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            Profiler.BeginSample("Encode Prompt(s)");
            var embeddings = EncodePrompt(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);
            Profiler.EndSample();

            // get the initial random noise unless the user supplied it
            TensorShape latentsShape = GetLatentsShape();
            if (latents == null) {
                Profiler.BeginSample("Generate Noise");
                latents = _ops.RandomNormal(latentsShape, 0, 1, seed.Value);
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
                Tensor<float> latentModelInput = doClassifierFreeGuidance ? _ops.Concatenate(latents, latents, 0) : latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                Profiler.BeginSample("Prepare Timestep Tensor");
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(batchSize), t);
                Profiler.EndSample();

                _ops.ExecuteCommandBufferAndClear();

                Profiler.BeginSample("Execute Unet");
                Tensor<float> noisePred = Unet.Execute(latentModelInput, timestep, embeddings.PromptEmbeds);
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
                var stepArgs = new Scheduler.StepArgs(noisePred, t, latents, eta, generator: generator);
                var schedulerOutput = Scheduler.Step(stepArgs);
                latents = schedulerOutput.PrevSample;
                Profiler.EndSample();

                if (callback != null) {
                    Profiler.BeginSample($"{GetType()} Callback");
                    _ops.ExecuteCommandBufferAndClear();
                    callback.Invoke(i / Scheduler.Order, t, latents);
                    Profiler.EndSample();
                }

                i++;
            }
            Profiler.EndSample();

            Profiler.BeginSample($"Scale Latents");
            Tensor<float> result = _ops.Div(latents, 0.18215f);
            Profiler.EndSample();

            // batch decode
            if (batchSize > 1) {
                throw new NotImplementedException();
            }

            _ops.ExecuteCommandBufferAndClear();

            Profiler.BeginSample($"VaeDecoder Decode Image");
            Tensor<float> outputImage = VaeDecoder.Execute(result);
            Profiler.EndSample();

            Profiler.BeginSample($"PostProcess Image");
            outputImage = ImageProcessor.PostProcess(outputImage, doDenormalize: true);
            Profiler.EndSample();

            Profiler.EndSample();
            return outputImage;
        }

        internal override Embeddings EncodePrompt(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            Tensor<float> promptEmbeds = null,
            Tensor<float> negativePromptEmbeds = null,
            Tensor<float> pooledPromptEmbeds = null,
            Tensor<float> negativePooledPromptEmbeds = null)
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
                using Tensor<int> textIdTensor = new Tensor<int>(new TensorShape(batchSize, textInputIds.Length), textInputIds);
                Profiler.EndSample();

                Profiler.BeginSample("Execute TextEncoder");
                TextEncoder.Execute(textIdTensor);
                Profiler.EndSample();

                promptEmbeds = TextEncoder.CopyOutput(0) as Tensor<float>;
                _ops.WaveOwnership(promptEmbeds);
            }

            promptEmbeds = _ops.Repeat(promptEmbeds, numImagesPerPrompt, axis: 0);

            // get unconditional embeddings for classifier free guidance
            if (doClassifierFreeGuidance && negativePromptEmbeds == null) {
                List<string> uncondTokens;
                if (negativePrompt == null) {
                    uncondTokens = Enumerable.Repeat("", batchSize).ToList();
                } else if (prompt.GetType() != negativePrompt.GetType()) {
                    throw new ArgumentException($"`negativePrompt` should be the same type as `prompt`, but got {negativePrompt.GetType()} != {prompt.GetType()}.");
                } else if (negativePrompt is SingleInput) {
                    uncondTokens = Enumerable.Repeat((negativePrompt as SingleInput).Text, batchSize).ToList();
                } else if (batchSize != (negativePrompt as BatchInput).Sequence.Count) {
                    throw new ArgumentException($"`negativePrompt`: {negativePrompt} has batch size {(negativePrompt as BatchInput).Sequence.Count}, " +
                        $"but `prompt`: {prompt} has batch size {batchSize}. Please make sure that passed `negativePrompt` matches " +
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
                int[] uncondInputIds = uncondInput.InputIds.ToArray() ?? throw new Exception("Failed to get unconditioned input ids.");
                Profiler.EndSample();

                Profiler.BeginSample("Prepare Unconditioned Text ID Tensor");
                using Tensor<int> uncondIdTensor = new Tensor<int>(new TensorShape(batchSize, uncondInputIds.Length), uncondInputIds);
                Profiler.EndSample();

                Profiler.BeginSample("Execute TextEncoder For Unconditioned Input");
                negativePromptEmbeds = TextEncoder.Execute(uncondIdTensor)[0] as Tensor<float>;
                Profiler.EndSample();
            }

            if (doClassifierFreeGuidance) {
                negativePromptEmbeds = _ops.Repeat(negativePromptEmbeds, numImagesPerPrompt, axis: 0);

                // For classifier free guidance, we need to do two forward passes.
                // Here we concatenate the unconditional and text embeddings into a single batch
                // to avoid doing two forward passes
                Profiler.BeginSample("Concat Prompt Embeds For Classifier-Fee Guidance");
                promptEmbeds = _ops.Concatenate(negativePromptEmbeds, promptEmbeds, 0);
                Profiler.EndSample();
            }

            _ops.ExecuteCommandBufferAndClear();
            return new Embeddings() { PromptEmbeds = promptEmbeds };
        }

        private TensorShape GetLatentsShape() {
            return new TensorShape(
                batchSize * numImagesPerPrompt,
                4, // unet.in_channels
                height / 8,
                width / 8
            );
        }

        public override void Dispose() {
            base.Dispose();
        }
    }
}
