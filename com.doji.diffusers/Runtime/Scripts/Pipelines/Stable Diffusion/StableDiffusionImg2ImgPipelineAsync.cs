using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Async Stable Diffusion Image to Image pipeline 
    /// </summary>
    public partial class StableDiffusionImg2ImgPipelineAsync : DiffusionPipelineAsync, IDisposable {

        public VaeEncoder VaeEncoder { get; protected set; }
        public VaeImageProcessor ImageProcessor { get; protected set; }

        private Ops _ops;

        /// <summary>
        /// Initializes a new async stable diffusion img2img pipeline.
        /// </summary>
        public StableDiffusionImg2ImgPipelineAsync(
            VaeEncoder vaeEncoder,
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            BackendType backend)
        {
            VaeEncoder = vaeEncoder;
            ImageProcessor = new VaeImageProcessor(/*vaeScaleFactor: self.vae_scale_factor*/);
            VaeDecoder = vaeDecoder;
            Tokenizer = tokenizer;
            TextEncoder = textEncoder;
            Scheduler = scheduler;
            Unet = unet;
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        public override Task<TensorFloat> GenerateAsync(
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
            throw new NotImplementedException($"This overload of the GenerateAsync() method is not implemented for the {GetType().Name}.");
        }

        /// <summary>
        /// Execute the pipeline asynchronously.
        /// </summary>
        /// <inheritdoc cref="Generate(Input, TensorFloat, float, int, float, Input, int, float, uint?, Action{int, float, TensorFloat})"/>

        public async Task<TensorFloat> GenerateAsync(
            Input prompt,
            TensorFloat image,
            float strength = 0.8f,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            Input negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            Action<int, float, TensorFloat> callback = null)
        {
            _prompt = prompt;
            _negativePrompt = negativePrompt;
            _steps = numInferenceSteps;
            _guidanceScale = guidanceScale;
            _numImagesPerPrompt = numImagesPerPrompt;
            _eta = eta;
            _seed = seed;
            CheckInputs();

            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            } else if (prompt is TextInput) {
                _batchSize = 1;
            } else if (prompt is BatchInput prompts) {
                _batchSize = prompts.Sequence.Count;
            } else {
                throw new ArgumentException($"Invalid prompt argument {nameof(prompt)}");
            }


            System.Random generator = null;
            if (_seed == null) {
                generator = new System.Random();
                _seed = unchecked((uint)generator.Next());
            }

            // set timesteps
            Scheduler.SetTimesteps(numInferenceSteps);

            //ImageProcessor.PreProcess(image);

            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            TensorFloat promptEmbeds = await EncodePromptAsync(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);

            // encode the init image into latents and scale the latents
            TensorFloat initLatents = VaeEncoder.Execute(image);
            initLatents = _ops.Mul(initLatents, VaeDecoder.Config.ScalingFactor ?? 0.18215f);

            if (_batchSize != initLatents.shape[0]) {
                throw new ArgumentException($"Mismatch between batch size ({_batchSize}) and latents dim 0 ({initLatents.shape[0]}");
            } else {
                initLatents = _ops.Repeat(initLatents, numImagesPerPrompt, axis: 0);
            }

            // get the original timestep using initTimestep
            int offset = Scheduler.Config.StepsOffset ?? 0;
            int initTimestep = (int)MathF.Floor(numInferenceSteps * strength) + offset;
            initTimestep = Math.Min(initTimestep, numInferenceSteps);

            float[] timesteps = Scheduler.GetTimestepsFromEnd(initTimestep);
            timesteps = timesteps.Repeat(_batchSize * numImagesPerPrompt);
            using TensorFloat timestepsT = new TensorFloat(new TensorShape(_batchSize * numImagesPerPrompt), timesteps);

            // add noise to latents using the timesteps
            var noise = _ops.RandomNormal(initLatents.shape, 0, 1, _seed);
            initLatents = Scheduler.AddNoise(initLatents, noise, timestepsT);

            _latents = initLatents;
            int tStart = Math.Max(numInferenceSteps - initTimestep + offset, 0);
            timesteps = Scheduler.GetTimesteps()[tStart..];

            int i = 0;
            foreach (float t in timesteps) {
                // expand the latents if doing classifier free guidance
                TensorFloat latentModelInput = doClassifierFreeGuidance ? _ops.Concatenate(_latents, _latents, 0) : _latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(_batchSize), t);

                TensorFloat noisePred = await Unet.ExecuteAsync(latentModelInput, timestep, promptEmbeds);

                // perform guidance
                if (doClassifierFreeGuidance) {
                    (var noisePredUncond, var noisePredText) = _ops.SplitHalf(noisePred, axis: 0);
                    var tmp = _ops.Sub(noisePredText, noisePredUncond);
                    var tmp2 = _ops.Mul(guidanceScale, tmp);
                    noisePred = _ops.Add(noisePredUncond, tmp2);
                }

                // compute the previous noisy sample x_t -> x_t-1
                var stepArgs = new Scheduler.StepArgs(noisePred, t, _latents, eta, generator: generator);
                var schedulerOutput = Scheduler.Step(stepArgs);
                _latents = schedulerOutput.PrevSample;

                if (callback != null) {
                    callback.Invoke(i / Scheduler.Order, t, _latents);
                }

                i++;
            }

            TensorFloat result = _ops.Div(_latents, 0.18215f);

            // batch decode
            if (_batchSize > 1) {
                throw new NotImplementedException();
            }

            TensorFloat outputImage = await VaeDecoder.ExecuteAsync(result);

            return outputImage;
        }

        private async Task<TensorFloat> EncodePromptAsync(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null)
        {
            if (promptEmbeds == null) {
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

                using TensorInt textIdTensor = new TensorInt(new TensorShape(_batchSize, textInputIds.Length), textInputIds);

                promptEmbeds = (await TextEncoder.ExecuteAsync(textIdTensor))[0] as TensorFloat;
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

                int maxLength = promptEmbeds.shape[1];
                var uncondInput = Tokenizer.Encode<BatchInput>(
                    text: uncondTokens,
                    padding: Padding.MaxLength,
                    maxLength: maxLength,
                    truncation: Truncation.LongestFirst
                ) as BatchEncoding;
                int[] uncondInputIds = uncondInput.InputIds as int[] ?? throw new Exception("Failed to get unconditioned input ids.");

                using TensorInt uncondIdTensor = new TensorInt(new TensorShape(_batchSize, uncondInputIds.Length), uncondInputIds);

                ownsPromptEmbeds = true;
                promptEmbeds.TakeOwnership();
                negativePromptEmbeds = (await TextEncoder.ExecuteAsync(uncondIdTensor))[0] as TensorFloat;
            }

            if (doClassifierFreeGuidance) {
                negativePromptEmbeds = _ops.Repeat(negativePromptEmbeds, numImagesPerPrompt, axis: 0);

                // For classifier free guidance, we need to do two forward passes.
                // Here we concatenate the unconditional and text embeddings into a single batch
                // to avoid doing two forward passes
                TensorFloat combinedEmbeddings = _ops.Concatenate(negativePromptEmbeds, promptEmbeds, 0);

                if (ownsPromptEmbeds) {
                    promptEmbeds.Dispose();
                }

                return combinedEmbeddings;
            }

            return promptEmbeds;
        }

        public override void Dispose() {
            base.Dispose();
            _ops?.Dispose();
            VaeEncoder?.Dispose();
        }
    }
}