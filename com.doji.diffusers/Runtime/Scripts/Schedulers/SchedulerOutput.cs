using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public struct SchedulerOutput {

        public TensorFloat PrevSample { get; internal set; }
        public TensorFloat PredOriginalSample { get; internal set; }

        public SchedulerOutput(TensorFloat prevSample) : this(prevSample, null) { }

        public SchedulerOutput(TensorFloat prevSample, TensorFloat predOriginalSample) {
            PrevSample = prevSample;
            PredOriginalSample = predOriginalSample;
        }
    }
}