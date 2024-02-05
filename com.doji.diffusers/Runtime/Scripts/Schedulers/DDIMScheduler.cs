using static Doji.AI.Diffusers.ArrayUtils;
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

            float[] betas = GetBetas();
        }

        /// <summary>
        /// Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        /// </summary>
        /// <remarks>
        /// TODO: Eventually use tensor ops for all this, but that's somewhat blocked
        /// by availability of a CumProd() implementation in <see cref="Ops"/>.
        /// </remarks>
        public static float[] RescaleZeroTerminalSnr(float[] betas) {
            // Convert betas to alphas_bar_sqrt
            float[] alphas = Sub(1f, betas);
            float[] alphasCumprod = alphas.CumProd();
            float[] alphasBarSqrt = alphasCumprod.Sqrt();

            // Store old values
            float alphasBarSqrt0 = alphasBarSqrt[0];
            float alphasBarSqrtT = alphasBarSqrt[^1];

            // Shift so the last timestep is zero
            alphasBarSqrt = alphasBarSqrt.Sub(alphasBarSqrtT);

            // Scale so the first timestep is back to the old value
            alphasBarSqrt = alphasBarSqrt.Mul(alphasBarSqrt0 / (alphasBarSqrt0 - alphasBarSqrtT));

            // Convert alphas_bar_sqrt to betas
            float[] alphasBar = alphasBarSqrt.Pow(2f);
            float[] alphasRevertCumProd = alphasBar[1..].Div(alphasBar[..^1]);
            alphas[0] = alphasBar[0];
            for (int i = 1; i < alphas.Length; i++) {
                alphas[i] = alphasRevertCumProd[i];
            }

            betas = Sub(1f, alphas);

            return betas;
        }

        public override void Dispose() {
            base.Dispose();
        }
    }
}