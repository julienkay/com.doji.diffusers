using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public interface IDiffusionPipeline : IDisposable {
        public Metadata GetMetadata();
        public Tensor<float> Generate(Parameters parameters);
    }

    public interface ITxt2ImgPipeline : IDiffusionPipeline { }

    public interface IImg2ImgPipeline : IDiffusionPipeline {

        /// <summary>
        /// Execute the img2img generation.
        /// </summary>
        /// <remarks>
        /// This is an overload for the most common generation parameters for convenience.
        /// For more control and advanced pipeline usage, pass parameters via the
        /// <see cref="Generate(Parameters)"/> method instead.
        /// </remarks>
        public Tensor<float> Generate(
            string prompt,
            Tensor<float> image,
            int? numInferenceSteps = null,
            float? guidanceScale = null,
            string negativePrompt = null,
            float? strength = null)
        {
            Parameters parameters = new Parameters() {
                Prompt = prompt,
                Image = image,
                NumInferenceSteps = numInferenceSteps,
                GuidanceScale = guidanceScale,
                NegativePrompt = negativePrompt,
                Strength = strength,
            };
            return Generate(parameters);
        }
    }
    public interface IControlnetPipeline : IDiffusionPipeline { }
}