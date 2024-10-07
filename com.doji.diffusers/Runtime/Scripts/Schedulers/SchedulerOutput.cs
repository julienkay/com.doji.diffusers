using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public struct SchedulerOutput {

        /// <summary>
        /// A <see cref="Tensor<float>"/> of shape `(batch_size, num_channels, height, width)` for images:
        /// Computed sample(x_{ t - 1}) of previous timestep. `prev_sample` should be used as next model
        /// input in the denoising loop.
        /// </summary>
        public Tensor<float> PrevSample { get; internal set; }

        /// <summary>
        /// A <see cref="Tensor<float>"/> of shape `(batch_size, num_channels, height, width)` for images):
        /// The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
        /// `pred_original_sample` can be used to preview progress or for guidance.
        /// </summary>
        public Tensor<float> PredOriginalSample { get; internal set; }

        public SchedulerOutput(Tensor<float> prevSample) : this(prevSample, null) { }

        public SchedulerOutput(Tensor<float> prevSample, Tensor<float> predOriginalSample) {
            PrevSample = prevSample;
            PredOriginalSample = predOriginalSample;
        }
    }
}