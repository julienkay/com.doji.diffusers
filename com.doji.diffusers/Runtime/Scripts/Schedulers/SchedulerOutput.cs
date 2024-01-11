namespace Doji.AI.Diffusers {

    public class SchedulerOutput {

        public float[] PrevSample { get; internal set; }

        public SchedulerOutput(float[] prevSample) {
            PrevSample = prevSample;
        }

    }
}