using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public struct SchedulerOutput {

        public TensorFloat PrevSample { get; internal set; }

        public SchedulerOutput(TensorFloat prevSample) {
            PrevSample = prevSample;
        }
    }
}