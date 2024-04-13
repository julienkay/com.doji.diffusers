using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion Image to Image pipeline 
    /// </summary>
    public partial class StableDiffusionImg2ImgPipeline : DiffusionPipeline, IImg2ImgPipeline, IDisposable {

        public VaeEncoder VaeEncoder { get; protected set; }

        /// <summary>
        /// Initializes a new stable diffusion img2img pipeline.
        /// </summary>
        public StableDiffusionImg2ImgPipeline(
            VaeEncoder vaeEncoder,
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            BackendType backend) : base(backend)
        {
            VaeEncoder = vaeEncoder;
            VaeDecoder = vaeDecoder;
            TextEncoder = textEncoder;
            Tokenizer = tokenizer;
            Scheduler = scheduler;
            Unet = unet;
            _ops = new Ops(backend);
            ImageProcessor = new VaeImageProcessor(backend: backend);
        }

        public override Parameters GetDefaultParameters() {
            return new Parameters() {
                Strength = 0.8f,
                NumInferenceSteps = 50,
                GuidanceScale = 7.5f,
                NegativePrompt = null,
                NumImagesPerPrompt = 1,
                Eta = 0.0f,
                Seed = null,
                Callback = null
            };
        }

        public override TensorFloat Generate(Parameters parameters) {
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

            if (strength < 0.0f || strength > 1.0f) {
                throw new ArgumentException($"The value of strength should be in [0.0, 1.0] but is {strength}");
            }

            System.Random generator = null;
            if (seed == null) {
                generator = new System.Random();
                seed = unchecked((uint)generator.Next());
            }

            // Prepare timesteps
            Profiler.BeginSample($"{Scheduler.GetType().Name}.SetTimesteps");
            Scheduler.SetTimesteps(numInferenceSteps);
            Profiler.EndSample();

            // Preprocess image
            Profiler.BeginSample($"Preprocess image");
            image = ImageProcessor.PreProcess(image);
            Profiler.EndSample();

            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            Profiler.BeginSample("Encode Prompt(s)");
            TensorFloat promptEmbeds = EncodePrompt(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);
            Profiler.EndSample();

            // encode the init image into latents and scale the latents
            TensorFloat initLatents = VaeEncoder.Execute(image);
            initLatents = _ops.Mul(initLatents, VaeDecoder.Config.ScalingFactor ?? 0.18215f);

            if (batchSize != initLatents.shape[0]) {
                throw new ArgumentException($"Mismatch between batch size ({batchSize}) and latents dim 0 ({initLatents.shape[0]}");
            } else {
                initLatents = _ops.Repeat(initLatents, numImagesPerPrompt, axis: 0);
            }

            // get the original timestep using initTimestep
            int offset = Scheduler.Config.StepsOffset ?? 0;
            int initTimestep = (int)MathF.Floor(numInferenceSteps * strength) + offset;
            initTimestep = Math.Min(initTimestep, numInferenceSteps);

            float[] timesteps = Scheduler.GetTimestepsFromEnd(initTimestep);
            timesteps = timesteps.Repeat(batchSize * numImagesPerPrompt);
            using TensorFloat timestepsT = new TensorFloat(new TensorShape(batchSize * numImagesPerPrompt), timesteps);

            // add noise to latents using the timesteps
            Profiler.BeginSample("Generate Noise");
            TensorFloat noise = _ops.RandomNormal(initLatents.shape, 0, 1, seed.Value);
            initLatents = Scheduler.AddNoise(initLatents, noise, timestepsT);
            Profiler.EndSample();

            latents = initLatents;
            int tStart = Math.Max(numInferenceSteps - initTimestep + offset, 0);
            timesteps = Scheduler.GetTimesteps()[tStart..];

            Profiler.BeginSample($"Denoising Loop");
            int i = 0;
            foreach (float t in timesteps) {
                // expand the latents if doing classifier free guidance
                TensorFloat latentModelInput = doClassifierFreeGuidance ? _ops.Concatenate(latents, latents, 0) : latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                Profiler.BeginSample("Prepare Timestep Tensor");
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(batchSize), t);
                Profiler.EndSample();

                Profiler.BeginSample("Execute Unet");
                TensorFloat noisePred = Unet.Execute(latentModelInput, timestep, promptEmbeds);
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
            if (batchSize > 1) {
                throw new NotImplementedException();
            }

            Profiler.BeginSample($"VaeDecoder Decode Image");
            TensorFloat outputImage = VaeDecoder.Execute(result);
            Profiler.EndSample();

            Profiler.BeginSample($"PostProcess Image");
            outputImage = ImageProcessor.PostProcess(outputImage, doDenormalize: true);
            Profiler.EndSample();

            Profiler.EndSample();
            return outputImage;
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
                using TensorInt textIdTensor = new TensorInt(new TensorShape(batchSize, textInputIds.Length), textInputIds);
                Profiler.EndSample();

                Profiler.BeginSample("Execute TextEncoder");
                promptEmbeds = TextEncoder.Execute(textIdTensor)[0] as TensorFloat;
                Profiler.EndSample();
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
                int[] uncondInputIds = uncondInput.InputIds as int[] ?? throw new Exception("Failed to get unconditioned input ids.");
                Profiler.EndSample();

                Profiler.BeginSample("Prepare Unconditioned Text ID Tensor");
                using TensorInt uncondIdTensor = new TensorInt(new TensorShape(batchSize, uncondInputIds.Length), uncondInputIds);
                Profiler.EndSample();

                promptEmbeds = _ops.Copy(promptEmbeds); // "take ownership"
                Profiler.BeginSample("Execute TextEncoder For Unconditioned Input");
                negativePromptEmbeds = TextEncoder.Execute(uncondIdTensor)[0] as TensorFloat;
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

                return combinedEmbeddings;
            }

            return promptEmbeds;
        }

        public override void Dispose() {
            base.Dispose();
            VaeEncoder?.Dispose();
        }
    }
}
