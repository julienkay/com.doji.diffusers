using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public interface IDiffusionPipeline : IDisposable {
        public Parameters GetParameters();
    }

    public interface ITxt2ImgPipeline : IDiffusionPipeline {

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
            Action<int, float, TensorFloat> callback = null);

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
            Action<int, float, TensorFloat> callback = null);

        public TensorFloat Generate(
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
            Action<int, float, TensorFloat> callback = null);
    }

    public interface IImg2ImgPipeline : IDiffusionPipeline {

        /// <summary>
        /// Execute the img2img generation on this pipeline
        /// </summary>
        /// <inheritdoc cref="DiffusionPipeline.Generate(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, float, TensorFloat})"/>
        /// <param name="image">tensor representing an image batch, that will be used as the starting point for the process.</param>
        /// <param name="strength">Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. The
        /// <paramref name="image"/> will be used as a starting point, adding more noise to it the larger the <paramref name="strength"/>.
        /// number of denoising steps depends on the amount of noise initially added. When <paramref name="strength"/> is 1, added noise will
        /// be maximum and the denoising process will run for the full number of iterations specified in <paramref name="numInferenceSteps"/>.
        /// A value of 1, therefore, essentially ignores <paramref name="image"/>.</param>
        /// <param name="numInferenceSteps"> The number of denoising steps. More denoising steps usually lead to a higher quality image
        /// at the expense of slower inference. This parameter will be modulated by <paramref name="strength"/>.</param>
        public TensorFloat Generate(
            Input prompt,
            TensorFloat image,
            float strength = 0.8f,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            Input negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            Action<int, float, TensorFloat> callback = null);
    }
}