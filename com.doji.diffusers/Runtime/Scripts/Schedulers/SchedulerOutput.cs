using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public struct SchedulerOutput {

        /// <summary>
        /// A <see cref="TensorFloat"/> of shape `(batch_size, num_channels, height, width)` for images:
        /// Computed sample(x_{ t - 1}) of previous timestep. `prev_sample` should be used as next model
        /// input in the denoising loop.
        /// </summary>
        public TensorFloat PrevSample { get; internal set; }
        public TensorFloat PredOriginalSample { get; internal set; }

        public SchedulerOutput(TensorFloat prevSample) : this(prevSample, null) { }

        public SchedulerOutput(TensorFloat prevSample, TensorFloat predOriginalSample) {
            PrevSample = prevSample;
            PredOriginalSample = predOriginalSample;
        }
    }
}