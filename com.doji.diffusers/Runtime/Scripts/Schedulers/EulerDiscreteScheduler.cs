using static Doji.AI.ArrayUtils;
using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public class EulerDiscreteScheduler : EulerScheduler {

        public EulerDiscreteScheduler(SchedulerConfig config, BackendType backend = BackendType.GPUCompute) : base(config, backend) {
            Config.NumTrainTimesteps ??= 1000;
            Config.BetaStart ??= 0.0001f;
            Config.BetaEnd ??= 0.02f;
            Config.BetaSchedule ??= Schedule.Linear;
            Config.TrainedBetas ??= null;
            Config.PredictionType ??= Prediction.Epsilon;
            Config.InterpolationType ??= Interpolation.Linear;
            Config.UseKarrasSigmas ??= false;
            Config.SigmaMin ??= null;
            Config.SigmaMax ??= null;
            Config.TimestepSpacing ??= Spacing.Linspace;
            Config.TimestepType ??= Timestep.Discrete;
            Config.StepsOffset ??= 0;
            Config.RescaleBetasZeroSnr ??= false;

            Initialize();

            // TODO: Support the full EDM scalings for all prediction types and timestep types
            if (TimestepType == Timestep.Continuous && PredictionType == Prediction.V_Prediction) {
                for (int i = 0; i < Sigmas.Length; i++) {
                    Timesteps[i] = 0.25f * MathF.Log(Sigmas[i]);
                }
            }
        }

        /// <inheritdoc/>
        public override void SetTimesteps(int numInferenceSteps) {
            base.SetTimesteps(numInferenceSteps);

            float[] log_sigmas = Sigmas.Log();
            //var log_sigmas = _ops.Log(sigmas);

            if (InterpolationType == Interpolation.Linear) {
                Sigmas = Interpolate(Timesteps, ArangeF(0, Sigmas.Length), Sigmas);
            } else if (InterpolationType == Interpolation.LogLinear) {
                Sigmas = Linspace(MathF.Log(Sigmas[^1]), MathF.Log(Sigmas[0]), NumInferenceSteps + 1).Exp();
            } else {
                throw new ArgumentException($"{InterpolationType} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Interpolation)))}.");
            }

            if (UseKarrasSigmas) {
                Sigmas = ConvertToKarras(Sigmas, NumInferenceSteps);
                for (int i = 0; i < Sigmas.Length; i++) {
                    Timesteps[i] = SigmaToT(Sigmas[i], log_sigmas);
                }
            }

            // TODO: Support the full EDM scalings for all prediction types and timestep types
            if (TimestepType == Timestep.Continuous && PredictionType == Prediction.V_Prediction) {
                for (int i = 0; i < Sigmas.Length; i++) {
                    Timesteps[i] = 0.25f * MathF.Log(Sigmas[i]);
                }
            } 

            Sigmas = Sigmas.Concatenate(0);
            StepIndex = null;
            BeginIndex = null;
        }

        public float SigmaToT(float sigma, float[] logSigmas) {
            using var ops = new Ops(BackendType.CPU);

            using Tensor<float> sigmaT = new Tensor<float>(new TensorShape(), new[] { sigma });
            using Tensor<float> logSigmasT = new Tensor<float>(new TensorShape(logSigmas.Length), logSigmas);
            using Tensor<float> zero = new Tensor<float>(new TensorShape(), new[] { 0f });

            // get log sigma
            float logSigma = MathF.Log(MathF.Max(sigma, 1e-10f));

            // get distribution
            var expanded = ops.Reshape(logSigmasT, new TensorShape(5, 1));
            var dists = ops.Sub(logSigmasT, expanded);

            // get sigmas range
            var greater = ops.GreaterOrEqual(dists, zero);
            var cumsum = ops.CumSum(greater, 0);
            var argmax = ops.ArgMax(cumsum, 0, true);
            var clip = ops.Clip(cumsum, 0, logSigmas.Length - 2);
            Debug.Assert(clip.shape.Equals(new TensorShape(1)));

            int lowIdx = clip[0];
            int highIdx = lowIdx + 1;

            float low = logSigmas[lowIdx];
            float high = logSigmas[highIdx];

            // interpolate sigmas
            float w = (low - logSigma) / (low - high);
            w = MathF.Min(MathF.Max(w, 0), 1);

            // transform interpolation to time range
            float t = (1f - w) * lowIdx + w * highIdx;

            return t;
        }

        /// <summary>
        /// Constructs the noise schedule of Karras et al. (2022).
        /// </summary>
        private float[] ConvertToKarras(float[] inSigmas, int numInferenceSteps) {
            // Hack to make sure that other schedulers which copy this function don't break
            // TODO: Add this logic to the other schedulers
            float sigmaMin = SigmaMin != null ? SigmaMin.Value : inSigmas[^1];
            float sigmaMax = SigmaMax != null ? SigmaMax.Value : inSigmas[0];

            float rho = 7.0f;  // 7.0 is the value used in the paper
            float[] ramp = Linspace(0, 1, numInferenceSteps);
            float minInvRho = MathF.Pow(sigmaMin, 1f / rho);
            float maxInvRho = MathF.Pow(sigmaMax, 1f / rho);
            float[] sigmas = new float[numInferenceSteps];

            for (int i = 0; i < numInferenceSteps; i++) {
                sigmas[i] = MathF.Pow(maxInvRho + ramp[i] * (minInvRho - maxInvRho), rho);
            }

            return sigmas;
        }

        /// <inheritdoc/>
        public override SchedulerOutput Step(StepArgs args) {
            SetStepArgs(args);

            if (!IsScaleInputCalled) {
                Debug.LogWarning("The `ScaleModelInput()` function should be called before `Step()`.");
            }

            if (StepIndex == null) {
                InitStepIndex(timestep);
            }

            var sigma = Sigmas[StepIndex.Value];

            var gamma = 0.0f;
            if (s_tmin <= sigma && sigma <= s_tmax) {
                gamma = MathF.Min(s_churn / (Sigmas.Length - 1), MathF.Sqrt(2f) - 1f);
            }

            generator ??= new System.Random();
            uint seed = unchecked((uint)generator.Next());
            var noise = Ops.RandomNormal(modelOutput.shape, 0, 1, seed);

            var eps = Ops.Mul(noise, s_noise);
            float sigmaHat = sigma * (gamma + 1f);

            if (gamma > 0f) {
                var tmp1 = Ops.Mul(eps, MathF.Pow(sigmaHat, 2f) - MathF.Pow(sigma, 2f));
                sample = Ops.Add(sample, tmp1);
            }

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            // NOTE: "original_sample" should not be an expected prediction_type but is left in for
            // backwards compatibility
            Tensor<float> predOriginalSample;
            if (PredictionType == Prediction.Sample) {
                predOriginalSample = modelOutput;
            } else if (PredictionType == Prediction.Epsilon) {
                predOriginalSample = Ops.Sub(sample, Ops.Mul(modelOutput, sigmaHat));
            } else if (PredictionType == Prediction.V_Prediction) {
                // denoised = model_output * c_out + input * c_skip
                predOriginalSample = Ops.Add(Ops.Mul(modelOutput, -sigma / MathF.Sqrt(MathF.Pow(sigma, 2f) + 1f)),
                    Ops.Div(sample, MathF.Pow(sigma, 2f) + 1f));
            } else {
                throw new ArgumentException($"Invalid prediction_type: {PredictionType}");
            }

            // 2. Convert to an ODE derivative
            Tensor<float> derivative = Ops.Div(Ops.Sub(sample, predOriginalSample), sigmaHat);
            float dt = Sigmas[StepIndex.Value + 1] - sigmaHat;
            Tensor<float> prevSample = Ops.Add(sample, Ops.Mul(derivative, dt));

            // Increase step index by one
            StepIndex++;

            return new SchedulerOutput(prevSample, predOriginalSample);
        }

        public static EulerDiscreteScheduler FromConfig(SchedulerConfig cfg, BackendType b) => FromConfig<EulerDiscreteScheduler>(cfg, b);
    }
}