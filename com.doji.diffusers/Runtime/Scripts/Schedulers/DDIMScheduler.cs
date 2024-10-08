﻿using static Doji.AI.ArrayUtils;
using Unity.Sentis;
using System;

namespace Doji.AI.Diffusers {

    public class DDIMScheduler : SchedulerInt {

        public float[] Betas { get; private set; }
        public float FinalAlphaCumprod { get; private set; }

        public DDIMScheduler(SchedulerConfig config = null, BackendType backend = BackendType.GPUCompute) : base(config, backend) {
            Config.NumTrainTimesteps        ??= 1000;
            Config.BetaStart                ??= 0.0001f;
            Config.BetaEnd                  ??= 0.02f;
            Config.BetaSchedule             ??= Schedule.Linear;
            Config.TrainedBetas             ??= null;
            Config.ClipSample               ??= true;
            Config.ClipSampleRange          ??= 1.0f;
            Config.SetAlphaToOne            ??= true;
            Config.StepsOffset              ??= 0;
            Config.PredictionType           ??= Prediction.Epsilon;
            Config.Thresholding             ??= false;
            Config.DynamicThresholdingRatio ??= 0.995f;
            Config.SampleMaxValue           ??= 1.0f;
            Config.TimestepSpacing          ??= Spacing.Leading;
            Config.RescaleBetasZeroSnr      ??= false;

            Betas = GetBetas();

            // Rescale for zero SNR
            if (RescaleBetasZeroSnr) {
                Betas = RescaleZeroTerminalSnr(Betas);
            }

            float[] alphas = Sub(1f, Betas);
            AlphasCumprodF = alphas.CumProd();
            AlphasCumprod = new Tensor<float>(new TensorShape(alphas.Length), AlphasCumprodF);

            // At every step in ddim, we are looking into the previous alphas_cumprod
            // For the final step, there is no previous alphas_cumprod because we are already at 0
            // `set_alpha_to_one` decides whether we set this parameter simply to one or
            // whether we use the final alpha of the "non-previous" one.
            FinalAlphaCumprod = SetAlphaToOne ? 1.0f : AlphasCumprodF[0];

            NumInferenceSteps = 0;
            Timesteps = Arange(0, NumTrainTimesteps).Reverse();
        }

        /// <summary>
        /// Rescales betas to have zero terminal SNR Based on <see href="https://arxiv.org/pdf/2305.08891.pdf"/> (Algorithm 1)
        /// </summary>
        /// <remarks>
        /// TODO: Eventually use tensor ops for all this, but that's somewhat blocked
        /// by availability of a CumProd() implementation in <see cref="Ops"/>.
        /// </remarks>
        internal static float[] RescaleZeroTerminalSnr(float[] betas) {
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

        /// <inheritdoc/>
        public override SchedulerOutput Step(StepArgs args) {
            SetStepArgs(args);

            if (NumInferenceSteps == 0) {
                throw new ArgumentException("Number of inference steps is '0', you need to run 'SetTimesteps' after creating the scheduler");
            }

            // See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
            // Ideally, read DDIM paper in-detail understanding

            // Notation (<variable name> -> <name in paper>
            // - pred_noise_t -> e_theta(x_t, t)
            // - pred_original_sample -> f_theta(x_t, t) or x_0
            // - std_dev_t -> sigma_t
            // - eta -> η
            // - pred_sample_direction -> "direction pointing to x_t"
            // - pred_prev_sample -> "x_t-1"

            // 1. get previous step value (=t-1)
            int prevTimestep = (int)timestep - NumTrainTimesteps / NumInferenceSteps;

            // 2. compute alphas, betas
            float alphaProdT = AlphasCumprodF[(int)timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? AlphasCumprodF[prevTimestep] : FinalAlphaCumprod;

            float betaProdT = 1.0f - alphaProdT;

            // 3. compute predicted original sample from predicted noise also called
            // "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            Tensor<float> predOriginalSample, predEpsilon;

            if (PredictionType == Prediction.Epsilon) {
                var tmp = Ops.Mul(modelOutput, MathF.Sqrt(betaProdT));
                var tmp2 = Ops.Sub(sample, tmp);
                predOriginalSample = Ops.Div(tmp2, MathF.Sqrt(alphaProdT));
                predEpsilon = modelOutput;
            } else if (PredictionType == Prediction.Sample) {
                predOriginalSample = modelOutput;
                var tmp = Ops.Mul(predOriginalSample, MathF.Sqrt(alphaProdT));
                var tmp2 = Ops.Sub(sample, tmp);
                predEpsilon = Ops.Div(tmp2, MathF.Sqrt(betaProdT));
            } else if (PredictionType == Prediction.V_Prediction) {
                var tmp = Ops.Mul(sample, MathF.Pow(alphaProdT, 0.5f));
                var tmp2 = Ops.Mul(modelOutput, MathF.Pow(betaProdT, 0.5f));
                predOriginalSample = Ops.Sub(tmp, tmp2);
                var tmp3 = Ops.Mul(modelOutput, MathF.Pow(alphaProdT, 0.5f));
                var tmp4 = Ops.Mul(sample, MathF.Pow(betaProdT, 0.5f));
                predEpsilon = Ops.Add(tmp3, tmp4);
            } else {
                throw new ArgumentException(
                    $"prediction_type given as {PredictionType} must be one of {string.Join(", ", Enum.GetNames(typeof(Prediction)))}."
                );
            }

            // 4. Clip or threshold "predicted x_0"
            if (Thresholding) {
                predOriginalSample = ThresholdSample(predOriginalSample);
            } else if (ClipSample) {
                predOriginalSample = Ops.Clip(predOriginalSample, -ClipSampleRange, ClipSampleRange);
            }

            // 5. compute variance: "sigma_t(η)" -> see formula (16)
            // σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            float variance = GetVariance((int)timestep, prevTimestep);
            float stdDevT = eta * MathF.Pow(variance, 0.5f);

            if (useClippedModelOutput) {
                // the predEpsilon is always re-derived from the clipped x_0 in Glide
                var tmp = Ops.Mul(predOriginalSample, MathF.Pow(alphaProdT, 0.5f));
                var tmp2 = Ops.Sub(sample, tmp);
                predEpsilon = Ops.Div(tmp2, MathF.Pow(betaProdT, 0.5f));
            }

            // 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var predSampleDirection = Ops.Mul(predEpsilon, MathF.Pow(1.0f - alphaProdTPrev - MathF.Pow(stdDevT, 2f), 0.5f));

            // 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var x_t = Ops.Mul(predOriginalSample, MathF.Pow(alphaProdTPrev, 0.5f));
            var prevSample = Ops.Add(x_t, predSampleDirection);

            if (eta > 0) {
                if (varianceNoise != null && generator != null) {
                    throw new ArgumentException(
                        "Cannot pass both generator and variance_noise. Please make sure that either `generator` or" +
                        " `variance_noise` stays `None`."
                    );
                }

                if (varianceNoise == null) {
                    int seed = generator.Next();
                    varianceNoise = Ops.RandomNormal(modelOutput.shape, 0f, 1f, seed);
                }
                var varianceTensor = Ops.Mul(varianceNoise, stdDevT);

                prevSample = Ops.Add(prevSample, varianceTensor);
            }

            return new SchedulerOutput(prevSample: prevSample, predOriginalSample: predOriginalSample);
        }

        /// <summary>
        /// "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0
        /// (the prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range[-s, s] and then
        /// divide by s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively
        /// preventing pixels from saturation at each step.We find that dynamic thresholding results in significantly
        /// better photorealism as well as better image-text alignment, especially when using very large guidance weights."
        /// <seealso href="https://arxiv.org/abs/2205.11487"/>
        /// </summary>
        private Tensor<float> ThresholdSample(Tensor<float> sample) {
            TensorShape shape = sample.shape;

            // Flatten sample for doing quantile calculation along each image
            sample.Reshape(sample.shape.Flatten());

            var absSample = Ops.Abs(sample);  // "a certain percentile absolute pixel value"

            var s = Ops.Quantile(absSample, DynamicThresholdingRatio, 1);
            s = Ops.Clip(s, 1, SampleMaxValue);  // When clamped to min=1, equivalent to standard clipping to [-1, 1]
            s.Reshape(s.shape.Unsqueeze(1));  // (batch_size, 1) because clamp will broadcast along dim=0
            var clip = Ops.Clamp(sample, Ops.Neg(s), s);  // "we threshold xt0 to the range [-s, s] and then divide by s"
            sample = Ops.Div(clip, s);

            sample.Reshape(shape);

            return sample;
        }

        private float GetVariance(int timestep, int prevTimestep) {
            float alphaProdT = AlphasCumprodF[timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? AlphasCumprodF[prevTimestep] : FinalAlphaCumprod;
            float betaProdT = 1.0f - alphaProdT;
            float betaProdTPrev = 1.0f - alphaProdTPrev;

            float variance = (betaProdTPrev / betaProdT) * (1.0f - alphaProdT / alphaProdTPrev);

            return variance;
        }

        public override void Dispose() {
            AlphasCumprod?.Dispose();
            base.Dispose();
        }

        public static DDIMScheduler FromConfig(SchedulerConfig cfg, BackendType b) => FromConfig<DDIMScheduler>(cfg, b);
    }
}