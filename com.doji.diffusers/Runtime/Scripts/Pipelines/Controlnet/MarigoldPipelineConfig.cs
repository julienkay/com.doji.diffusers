using Newtonsoft.Json;

namespace Doji.AI.Diffusers {

    public partial class MarigoldPipelineConfig : PipelineConfig {

        /// <summary>
        /// A model property specifying whether the predicted depth maps are scale-invariant.
        /// This value must be set in the model config. When used together with the
        /// `shift_invariant=True` flag, the model is also called "affine-invariant".
        /// </summary>
        [JsonProperty("scale_invariant")]
        public bool ScaleInvariant { get; set; }

        /// <summary>
        /// A model property specifying whether the predicted depth maps are shift-invariant.
        /// This value must be set in the model config. When used together with the
        /// `scale_invariant=True` flag, the model is also called "affine-invariant".
        /// </summary>
        [JsonProperty("shift_invariant")]
        public bool ShiftInvariant { get; set; }

        /// <summary>
        /// The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
        /// quality with the given model.This value must be set in the model config.When the pipeline is called
        /// without explicitly setting `num_inference_steps`, the default value is used.This is required to ensure
        /// reasonable results with various model flavors compatible with the pipeline, such as those relying on very
        /// short denoising schedules(`LCMScheduler`) and those with full diffusion schedules(`DDIMScheduler`).
        /// </summary>
        [JsonProperty("default_denoising_steps")]
        public int? DefaultDenoisingSteps { get; set; }

        /// <summary>
        /// The recommended value of the `processing_resolution` parameter of the pipeline.
        /// This value must be set in the model config. When the pipeline is called without
        /// explicitly setting `processing_resolution`, the default value is used. This is
        /// required to ensure reasonable results with various model flavors trained wit
        /// varying optimal processing resolution values.
        /// </summary>
        [JsonProperty("default_processing_resolution")]
        public int? DefaultProcessingResolution { get; set; }
    }
}
