using Unity.Sentis;

namespace Doji.AI.Diffusers {
    public class DDIMScheduler : Scheduler {

        public DDIMScheduler(
            SchedulerConfig config,
            Prediction predictionType = Prediction.Epsilon,
            Spacing timestepSpacing = Spacing.Leading,
            BackendType backend = BackendType.GPUCompute) : base(backend)
        {
            Config = config ?? new SchedulerConfig() {
                NumTrainTimesteps = 1000,
                BetaStart = 0.0001f,
                BetaEnd = 0.02f,
                BetaSchedule = Schedule.Linear,
                TrainedBetas = null,
                SkipPrkSteps = false,
                SetAlphaToOne = false,
                StepsOffset = 0,
            };

        }

        public override void Dispose() {
            base.Dispose();
        }
    }
}