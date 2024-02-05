using Unity.Sentis;

namespace Doji.AI.Diffusers {
    public class DDIMScheduler : Scheduler {

        public DDIMScheduler(
            SchedulerConfig config,
            BackendType backend = BackendType.GPUCompute) : base(backend)
        {
            Config = config ?? new SchedulerConfig() {
                NumTrainTimesteps = 1000,
                BetaStart = 0.0001f,
                BetaEnd = 0.02f,
                BetaSchedule = Schedule.Linear,
                TrainedBetas = null,
                ClipSample = true,
                SetAlphaToOne = true,
                StepsOffset = 0,
                PredictionType = Prediction.Epsilon,
                Thresholding = false,
                DynamicThresholdingRatio = 0.995f,
                ClipSampleRange = 1.0f,
                SampleMaxValue = 1.0f,
                TimestepSpacing = Spacing.Leading,
                RescaleBetasZeroSnr = false
            };

        }

        public override void Dispose() {
            base.Dispose();
        }
    }
}