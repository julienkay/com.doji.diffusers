using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public abstract partial class DiffusionPipeline : IDisposable {

        public string NameOrPath { get; protected set; }
        public PipelineConfig Config { get; protected set; }

        public VaeDecoder VaeDecoder { get; protected set; }
        public ClipTokenizer Tokenizer { get; protected set; }
        public TextEncoder TextEncoder { get; protected set; }
        public Scheduler Scheduler { get; protected set; }
        public Unet Unet { get; protected set; }

        protected Input _prompt;
        protected Input _negativePrompt;
        protected int _steps;
        protected int _height;
        protected int _width;
        protected int _batchSize;
        protected int _numImagesPerPrompt;
        protected float _guidanceScale;
        protected float? _eta;
        protected uint? _seed;
        protected TensorFloat _latents;

        protected void CheckInputs() {
            if (_height % 8 != 0 || _width % 8 != 0) {
                throw new ArgumentException($"`height` and `width` have to be divisible by 8 but are {_height} and {_width}.");
            }
            if (_numImagesPerPrompt > 1) {
                throw new ArgumentException($"More than one image per prompt not supported yet. `numImagesPerPrompt` was {_numImagesPerPrompt}.");
            }
            if (_latents != null && _seed != null) {
                throw new ArgumentException($"Both a seed and pre-generated noise has been passed. Please use either one or the other.");
            }
        }

        public Parameters GetParameters() {
            if (_prompt is not SingleInput) {
                throw new NotImplementedException("GetParameters not yet implemented for batch inputs.");
            }

            return new Parameters() {
                PackageVersion = System.Diagnostics.FileVersionInfo.GetVersionInfo(System.Reflection.Assembly.GetExecutingAssembly().Location).ProductVersion,
                Prompt = (_prompt as SingleInput).Text,
                Model = NameOrPath,
                NegativePrompt = _negativePrompt != null ? (_negativePrompt as SingleInput).Text : null,
                Steps = _steps,
                Sampler = Scheduler.GetType().Name,
                CfgScale = _guidanceScale,
                Seed = _seed,
                Width = _width,
                Height = _height,
                Eta = _eta
            };
        }

        /// <inheritdoc cref="Generate(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, float, TensorFloat})"/>
        public TensorFloat Generate(
            string prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            string negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, float, TensorFloat> callback = null)
        {
            return Generate((TextInput)prompt, height, width, numInferenceSteps, guidanceScale,
               (TextInput)negativePrompt, numImagesPerPrompt, eta, seed, latents, callback);
        }

        /// <param name="prompt">The prompts used to generate the batch of images for.</param>
        /// <inheritdoc cref="Generate(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, float, TensorFloat})"/>
        public TensorFloat Generate(
            List<string> prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            List<string> negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, float, TensorFloat> callback = null)
        {
            return Generate((BatchInput)prompt, height, width, numInferenceSteps, guidanceScale,
                (BatchInput)negativePrompt, numImagesPerPrompt, eta, seed, latents, callback);
        }

        /// <summary>
        /// Execute the pipeline to generate images.
        /// </summary>
        /// <param name="prompt">The prompt or prompts to guide the image generation.
        /// If not defined, one has to pass `prompt_embeds` instead.</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="numInferenceSteps"> The number of denoising steps.
        /// More denoising steps usually lead to a higher quality image
        /// at the expense of slower inference.</param>
        /// <param name="guidanceScale">Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
        /// `guidance_scale` is defined as `w` of equation 2. of[Imagen Paper] (https://arxiv.org/pdf/2205.11487.pdf).
        /// Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
        /// that are closely linked to the text `prompt`, usually at the expense of lower image quality.</param>
        /// <param name="negativePrompt">The prompt or prompts not to guide the image generation.
        /// Ignored when not using guidance (i.e., ignored if <paramref name="guidanceScale"/> is less than `1`).</param>
        /// <param name="numImagesPerPrompt">The number of images to generate per prompt.</param>
        /// <param name="eta">Corresponds to parameter eta in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
        /// <see cref="DDIMScheduler"/>, will be ignored for others.</param>
        /// <param name="seed">A seed to use to generate initial noise. Set this to make generation deterministic.</param>
        /// <param name="latents">Pre-generated noise, sampled from a Gaussian distribution, to be used as inputs for image
        /// generation. If not provided, a latents tensor will be generated for you using the supplied <paramref name="seed"/>.</param>
        /// <param name="callback">A function that will be called at every step during inference.
        /// The function will be called with the following arguments:
        /// `callback(step: int, timestep: float, latents: TensorFloat)`.</param>
        public abstract TensorFloat Generate(
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
            Action<int, float, TensorFloat> callback = null
        );

        /// <inheritdoc cref="GenerateAsync(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, float, TensorFloat})"/>
        public async Task<TensorFloat> GenerateAsync(
            string prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            string negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, float, TensorFloat> callback = null)
        {
            return await GenerateAsync((TextInput)prompt, height, width, numInferenceSteps, guidanceScale,
               (TextInput)negativePrompt, numImagesPerPrompt, eta, seed, latents, callback);
        }

        /// <param name="prompt">The prompts used to generate the batch of images for.</param>
        /// <inheritdoc cref="GenerateAsync(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, float, TensorFloat})"/>
        public async Task<TensorFloat> GenerateAsync(
            List<string> prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            List<string> negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, float, TensorFloat> callback = null)
        {
            return await GenerateAsync((BatchInput)prompt, height, width, numInferenceSteps, guidanceScale,
                (BatchInput)negativePrompt, numImagesPerPrompt, eta, seed, latents, callback);
        }

        /// <summary>
        /// Execute the pipeline asynchronously.
        /// </summary>
        /// <inheritdoc cref="Generate(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, float, TensorFloat})"/>
        public abstract Task<TensorFloat> GenerateAsync(
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
            Action<int, float, TensorFloat> callback = null
        );

        public virtual void Dispose() {
            VaeDecoder?.Dispose();
            TextEncoder?.Dispose();
            Scheduler?.Dispose();
            Unet?.Dispose();
        }
    }
}