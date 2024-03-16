using Doji.AI.Transformers;
using System.Collections.Generic;
using System.Threading.Tasks;
using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public abstract class DiffusionPipelineAsync : DiffusionPipelineBase {

        /// <summary>
        /// Conversion to async pipelines
        /// </summary>
        public static implicit operator DiffusionPipeline(DiffusionPipelineAsync pipe) {
            if (pipe == null)
                throw new ArgumentNullException(nameof(pipe));
            if (pipe is StableDiffusionPipelineAsync) {
                return (StableDiffusionPipeline)(pipe as StableDiffusionPipelineAsync);
            } else if (pipe is StableDiffusionImg2ImgPipelineAsync) {
                return (StableDiffusionImg2ImgPipeline)(pipe as StableDiffusionImg2ImgPipelineAsync);
            } else if (pipe is StableDiffusionXLPipelineAsync) {
                return (StableDiffusionXLPipeline)(pipe as StableDiffusionXLPipelineAsync);
            } else {
                throw new InvalidCastException($"Cannot convert {pipe.GetType()} to DiffusionPipeline");
            }
        }

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
        /// <inheritdoc cref="DiffusionPipeline.Generate(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, float, TensorFloat})"/>
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
    }
}