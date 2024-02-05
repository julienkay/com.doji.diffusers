using static Doji.AI.Diffusers.ArrayUtils;
using Unity.Sentis;
using System;

namespace Doji.AI.Diffusers {

    public class DDIMScheduler : Scheduler {

        public TensorFloat AlphasCumprod { get; private set; }
        public float FinalAlphaCumprod { get; private set; }
        public int[] Timesteps { get; private set; }

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

            // Rescale for zero SNR
            if (RescaleBetasZeroSnr) {
                betas = RescaleZeroTerminalSnr(betas);
            }

            float[] alphas = Sub(1f, betas);
            float[] alphasCumprod = alphas.CumProd();
            AlphasCumprod = new TensorFloat(new TensorShape(alphas.Length), alphasCumprod);

            // At every step in ddim, we are looking into the previous alphas_cumprod
            // For the final step, there is no previous alphas_cumprod because we are already at 0
            // `set_alpha_to_one` decides whether we set this parameter simply to one or
            // whether we use the final alpha of the "non-previous" one.
            FinalAlphaCumprod = SetAlphaToOne ? 1.0f : alphasCumprod[0];

            NumInferenceSteps = 0;
            Timesteps = Arange(0, NumTrainTimesteps).Reverse();
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

        public override void SetTimesteps(int numInferenceSteps) {
            if (numInferenceSteps > NumTrainTimesteps) {
                throw new ArgumentException($"`num_inference_steps`: {numInferenceSteps} cannot be larger than " +
                    $"`self.config.train_timesteps`: {NumTrainTimesteps} as the unet model trained with this " +
                    $"scheduler can only handle maximal {NumTrainTimesteps} timesteps.");
            }
            NumInferenceSteps = numInferenceSteps;

            // "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if (TimestepSpacing == Spacing.Linspace) {
                Timesteps = GetTimeStepsLinspace().Reverse();
            } else if (TimestepSpacing == Spacing.Leading) {
                Timesteps = GetTimeStepsLeading().Reverse();
            } else if (TimestepSpacing == Spacing.Trailing) {
                Timesteps = GetTimeStepsTrailing().Reverse();
            } else {
                throw new ArgumentException($"{TimestepSpacing} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Spacing)))}.");
            }
        }

        public override void Dispose() {
            base.Dispose();
        }
    }
}