using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public abstract partial class DiffusionPipeline : DiffusionPipelineBase {

        /// <summary>
        /// Conversion to async pipelines
        /// </summary>
        public static implicit operator DiffusionPipelineAsync(DiffusionPipeline pipe) {
            if (pipe == null) throw new ArgumentNullException(nameof(pipe));
            if (pipe is StableDiffusionPipeline) {
                return (StableDiffusionPipelineAsync)(pipe as StableDiffusionPipeline);
            } else if (pipe is StableDiffusionImg2ImgPipeline) {
                return (StableDiffusionImg2ImgPipelineAsync)(pipe as StableDiffusionImg2ImgPipeline);
            } else if (pipe is StableDiffusionXLPipeline) {
                return (StableDiffusionXLPipelineAsync)(pipe as StableDiffusionXLPipeline);
            } else {
                throw new InvalidCastException($"Cannot convert {pipe.GetType()} to DiffusionPipelineAsync");
            }
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
        /// <param name="height">The height in pixels of the generated image.</param>
        /// <param name="width">The width in pixels of the generated image.</param>
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
    }
}