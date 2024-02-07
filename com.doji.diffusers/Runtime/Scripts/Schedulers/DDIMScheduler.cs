using static Doji.AI.Diffusers.ArrayUtils;
using Unity.Sentis;
using System;

namespace Doji.AI.Diffusers {

    public class DDIMScheduler : Scheduler {

        public TensorFloat AlphasCumprod { get; private set; }
        public float FinalAlphaCumprod { get; private set; }

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

        protected override SchedulerOutput Step(TensorFloat modelOutput, int timestep, TensorFloat sample) {
            throw new InvalidOperationException();
        }

        /// <inheritdoc/>
        public override SchedulerOutput Step(
            TensorFloat modelOutput,
            int timestep,
            TensorFloat sample,
            float eta = 0.0f,
            bool useClippedModelOutput = false,
            System.Random generator = null,
            TensorFloat varianceNoise = null)
        {
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
            int prevTimestep = timestep - NumTrainTimesteps / NumInferenceSteps;

            // 2. compute alphas, betas
            float alphaProdT = AlphasCumprod[timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? AlphasCumprod[prevTimestep] : FinalAlphaCumprod;

            float betaProdT = 1.0f - alphaProdT;

            // 3. compute predicted original sample from predicted noise also called
            // "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            TensorFloat predOriginalSample, predEpsilon;

            if (PredictionType == Prediction.Epsilon) {
                //predOriginalSample = (sample - _ops.Pow(betaProdT, 0.5f) * modelOutput) / (float)Math.Pow(alphaProdT, 0.5f);
                //predEpsilon = modelOutput;
                throw new NotImplementedException();
            } else if (PredictionType == Prediction.Sample) {
                //predOriginalSample = modelOutput;
                //predEpsilon = (sample - (float)MathF.Pow(alphaProdT, 0.5f) * predOriginalSample) / (float)Math.Pow(betaProdT, 0.5f);
                throw new NotImplementedException();
            } else if (PredictionType == Prediction.V_Prediction) {
                var tmp = _ops.Mul(sample, MathF.Pow(alphaProdT, 0.5f));
                var tmp2 = _ops.Mul(modelOutput, MathF.Pow(betaProdT, 0.5f));
                predOriginalSample = _ops.Sub(tmp, tmp2);
                var tmp3 = _ops.Mul(modelOutput, MathF.Pow(alphaProdT, 0.5f));
                var tmp4 = _ops.Mul(sample, MathF.Pow(betaProdT, 0.5f));
                predEpsilon = _ops.Add(tmp3, tmp4);
            } else {
                throw new ArgumentException(
                    $"prediction_type given as {PredictionType} must be one of {string.Join(", ", Enum.GetNames(typeof(Prediction)))}."
                );
            }

            // 4. Clip or threshold "predicted x_0"
            if (Thresholding) {
                predOriginalSample = ThresholdSample(predOriginalSample);
            } else if (ClipSample) {
                predOriginalSample = _ops.Clip(predOriginalSample, -ClipSampleRange, ClipSampleRange);
            }

            // 5. compute variance: "sigma_t(η)" -> see formula (16)
            // σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            float variance = GetVariance(timestep, prevTimestep);
            float stdDevT = eta * MathF.Pow(variance, 0.5f);

            if (useClippedModelOutput) {
                // the predEpsilon is always re-derived from the clipped x_0 in Glide
                var tmp = _ops.Mul(predOriginalSample, MathF.Pow(alphaProdT, 0.5f));
                var tmp2 = _ops.Sub(sample, tmp);
                predEpsilon = _ops.Div(tmp2, MathF.Pow(betaProdT, 0.5f));
            }

            // 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var predSampleDirection = _ops.Mul(predEpsilon, MathF.Pow(1.0f - alphaProdTPrev - MathF.Pow(stdDevT, 2f), 0.5f));

            // 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var x_t = _ops.Mul(predOriginalSample, MathF.Pow(alphaProdTPrev, 0.5f));
            var prevSample = _ops.Add(x_t, predSampleDirection);

            if (eta > 0) {
                if (varianceNoise != null && generator != null) {
                    throw new ArgumentException(
                        "Cannot pass both generator and variance_noise. Please make sure that either `generator` or" +
                        " `variance_noise` stays `None`."
                    );
                }

                if (varianceNoise == null) {
                    int seed = generator.Next();
                    varianceNoise = _ops.RandomNormal(modelOutput.shape, 0, 1, seed);
                }
                var varianceTensor = _ops.Mul(varianceNoise, stdDevT);

                prevSample = _ops.Add(prevSample, varianceTensor);
            }

            return new SchedulerOutput(prevSample: prevSample, predOriginalSample: predOriginalSample);
        }

        /// <summary>
        /// "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0
        /// (the prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range[-s, s] and then
        /// divide by s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively
        /// preventing pixels from saturation at each step.We find that dynamic thresholding results in significantly
        /// better photorealism as well as better image-text alignment, especially when using very large guidance weights."
        /// https://arxiv.org/abs/2205.11487
        /// </summary>
        private TensorFloat ThresholdSample(TensorFloat sample) {
            TensorShape shape = sample.shape;

            // Flatten sample for doing quantile calculation along each image
            sample = sample.ShallowReshape(sample.shape.Flatten()) as TensorFloat;

            var absSample = _ops.Abs(sample);  // "a certain percentile absolute pixel value"

            var s = _ops.Quantile(absSample, DynamicThresholdingRatio, 1);
            s = _ops.Clip(s, 1, SampleMaxValue);  // When clamped to min=1, equivalent to standard clipping to [-1, 1]
            s = s.ShallowReshape(s.shape.Unsqueeze(1)) as TensorFloat;  // (batch_size, 1) because clamp will broadcast along dim=0
            var clip = _ops.Clamp(sample, _ops.Neg(s), s);  // "we threshold xt0 to the range [-s, s] and then divide by s"
            sample = _ops.Div(clip, s);

            sample = sample.ShallowReshape(shape) as TensorFloat;

            return sample;
        }

        private float GetVariance(int timestep, int prevTimestep) {
            float alphaProdT = AlphasCumprod[timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? AlphasCumprod[prevTimestep] : FinalAlphaCumprod;
            float betaProdT = 1.0f - alphaProdT;
            float betaProdTPrev = 1.0f - alphaProdTPrev;

            float variance = (betaProdTPrev / betaProdT) * (1.0f - alphaProdT / alphaProdTPrev);

            return variance;
        }

        public override void Dispose() {
            AlphasCumprod?.Dispose();
            base.Dispose();
        }
    }
}