using static Doji.AI.Diffusers.ArrayUtils;
using System;
using Unity.Sentis;
using System.Collections.Generic;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public class EulerDiscreteScheduler : SchedulerFloat {

        public float[] Betas { get; private set; }
        public TensorFloat AlphasCumprod { get; private set; }
        private float[] _AlphasCumprod { get; set; }
        public TensorFloat SigmasT { get; protected set; }
        private float[] Sigmas { get; set; }

        public bool IsScaleInputCalled { get; private set; }

        public override float InitNoiseSigma {
            get {
                // TODO: can we calculate this once after calling SetTimesteps() and cache it?
                // standard deviation of the initial noise distribution
                var maxSigma = Sigmas.Max();
                if (TimestepSpacing == Spacing.Linspace || TimestepSpacing == Spacing.Trailing) {
                    return maxSigma;
                }
                return MathF.Pow(MathF.Pow(maxSigma, 2f) + 1, 0.5f);
            }
        }

        /// <summary>
        /// The index counter for current timestep. It will increae 1 after each scheduler step.
        /// </summary>
        private int? StepIndex { get; set; }

        /// <summary>
        /// The index for the first timestep. It should be set from pipeline before the inference.
        /// </summary>
        public int? BeginIndex { get; private set; }

        public EulerDiscreteScheduler(SchedulerConfig config, BackendType backend) : base(config, backend) {
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

            Betas = GetBetas();

            if (RescaleBetasZeroSnr) {
                Betas = DDIMScheduler.RescaleZeroTerminalSnr(Betas);
            }

            float[] alphas = Sub(1f, Betas);
            _AlphasCumprod = alphas.CumProd();
            AlphasCumprod = new TensorFloat(new TensorShape(alphas.Length), _AlphasCumprod);

            if (RescaleBetasZeroSnr) {
                // Close to 0 without being 0 so first sigma is not inf
                // FP16 smallest positive subnormal works well here
                //AlphasCumprod[^1] = MathF.Pow(2f, -24f);
                throw new NotImplementedException();
            }

            float[] tmp1 = Sub(1f, _AlphasCumprod);
            float[] tmp2 = tmp1.Div(_AlphasCumprod);
            float[] sigmas = tmp2.Pow(0.5f).Reverse();
            Timesteps = Linspace(0f, NumTrainTimesteps - 1, NumTrainTimesteps).Reverse();

            // setable values
            NumInferenceSteps = 0;

            // TODO: Support the full EDM scalings for all prediction types and timestep types
            if (TimestepType == Timestep.Continuous && PredictionType == Prediction.V_Prediction) {
                for (int i = 0; i < sigmas.Length; i++) {
                    Timesteps[i] = 0.25f * MathF.Log(sigmas[i]);
                }
            }

            sigmas = sigmas.Concatenate(0);

            IsScaleInputCalled = false;
            StepIndex = null;
            BeginIndex = null;
            SigmasT = new TensorFloat(new TensorShape(sigmas.Length), sigmas);
        }

        /// <summary>
        /// Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        /// </summary>
        /// <inheritdoc/>
        public override TensorFloat ScaleModelInput(TensorFloat sample, float timestep) {
            if (StepIndex == null) {
                InitStepIndex(timestep);
            }

            float sigma = Sigmas[StepIndex.Value];
            sample = _ops.Div(sample, MathF.Pow((MathF.Pow(sigma, 2f) + 1f), 0.5f));

            IsScaleInputCalled = true;
            return sample;
        }

        /// <inheritdoc/>
        public override void SetTimesteps(int numInferenceSteps) {
            NumInferenceSteps = numInferenceSteps;

            float[] timesteps;
            // "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if (TimestepSpacing == Spacing.Linspace) {
                timesteps = GetTimeStepsLinspaceF().Reverse();
            } else if (TimestepSpacing == Spacing.Leading) {
                timesteps = GetTimeStepsLeadingF().Reverse();
            } else if (TimestepSpacing == Spacing.Trailing) {
                timesteps = GetTimeStepsTrailingF().Reverse();
            } else {
                throw new ArgumentException($"{TimestepSpacing} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Spacing)))}.");
            }

            float[] tmp1 = Sub(1f, _AlphasCumprod);
            float[] tmp2 = tmp1.Div(_AlphasCumprod);
            float[] sigmas = tmp2.Pow(0.5f);
            float[] log_sigmas = sigmas.Log();
            //var tmp1 = _ops.Sub(1f, AlphasCumprod);
            //var tmp2 = _ops.Div(tmp1, AlphasCumprod);
            //using TensorFloat pow = new TensorFloat(0.5f);
            //var sigmas = _ops.Pow(tmp2, pow);
            //var log_sigmas = _ops.Log(sigmas);
            
            if (InterpolationType == Interpolation.Linear) {
                sigmas = Interpolate(Timesteps, ArangeF(0, NumTrainTimesteps), sigmas);
            } else  if (InterpolationType == Interpolation.LogLinear) {
                sigmas = Linspace(MathF.Log(sigmas[^1]), MathF.Log(sigmas[0]), NumInferenceSteps + 1).Exp();
            } else {
                throw new ArgumentException($"{InterpolationType} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Interpolation)))}.");
            }

            if (UseKarrasSigmas) {
                sigmas = ConvertToKarras(sigmas, NumInferenceSteps);
                for (int i = 0; i < sigmas.Length; i++) {
                    timesteps[i] = SigmaToT(sigmas[i], log_sigmas);
                }
            }

            // TODO: Support the full EDM scalings for all prediction types and timestep types
            if (TimestepType == Timestep.Continuous && PredictionType == Prediction.V_Prediction) {
                for (int i = 0; i < sigmas.Length; i++) {
                    Timesteps[i] = 0.25f * MathF.Log(sigmas[i]);
                }
            } else {
                Timesteps = timesteps;
            }

            Sigmas = sigmas.Concatenate(0);
            StepIndex = null;
            BeginIndex = null;
        }

        public float SigmaToT(float sigma, float[] logSigmas) {
            using TensorFloat sigmaT = new TensorFloat(sigma);
            using TensorFloat logSigmasT = new TensorFloat(new TensorShape(logSigmas.Length), logSigmas);
            using TensorFloat zero = new TensorFloat(0f);

            // get log sigma
            float logSigma = MathF.Log(MathF.Max(sigma, 1e-10f));

            // get distribution
            var expanded = _ops.Reshape(logSigmasT, new TensorShape(5, 1));
            var dists = _ops.Sub(logSigmasT, expanded);

            // get sigmas range
            var greater = _ops.GreaterOrEqual(dists, zero);
            var cumsum = _ops.CumSum(greater, 0);
            var argmax = _ops.ArgMax(cumsum, 0, true);
            var clip = _ops.Clip(cumsum, 0, logSigmas.Length - 2);
            Debug.Assert(clip.shape.Equals(new TensorShape(1)));
            clip.MakeReadable();

            int lowIdx = clip.ToReadOnlyArray()[0];
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

        private int IndexForTimestep(float timestep, float[] scheduleTimesteps = null) {
            scheduleTimesteps ??= Timesteps;

            List<int> indices = new List<int>();
            for (int i = 0; i < scheduleTimesteps.Length; i++) {
                if (scheduleTimesteps[i] == timestep) {
                    indices.Add(i);
                }
            }

            // The sigma index that is taken for the **very** first `step`
            // is always the second index (or the last index if there is only 1)
            // This way we can ensure we don't accidentally skip a sigma in
            // case we start in the middle of the denoising schedule (e.g. for image-to-image)
            int pos = indices.Count > 1 ? 1 : 0;

            return indices[pos];
        }

        private void InitStepIndex(float timestep) {
            if (BeginIndex == null) {
                StepIndex = IndexForTimestep(timestep);
            } else {
                StepIndex = BeginIndex;
            }
        }

        /// <inheritdoc cref="Step(TensorFloat, float, TensorFloat)"/>
        private SchedulerOutput Step(
            TensorFloat modelOutput,
            float timestep,
            TensorFloat sample,
            float s_churn = 0.0f,
            float s_tmin = 0.0f,
            float s_tmax = float.PositiveInfinity,
            float s_noise = 1.0f,
            uint? seed = null,
            TensorFloat varianceNoise = null)
        {
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

            var noise = _ops.RandomNormal(modelOutput.shape, 0, 1, seed.Value);

            var eps = _ops.Mul(noise, s_noise);
            float sigma_hat = sigma * (gamma + 1f);

            if (gamma > 0f) {
                var tmp1 = _ops.Mul(eps, MathF.Pow(sigma_hat, 2f) - MathF.Pow(sigma, 2f));
                sample = _ops.Add(sample, tmp1);
            }

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            // NOTE: "original_sample" should not be an expected prediction_type but is left in for
            // backwards compatibility
            TensorFloat predOriginalSample;
            if (PredictionType == Prediction.Sample) {
                predOriginalSample = modelOutput;
            } else if (PredictionType == Prediction.Epsilon) {
                predOriginalSample = _ops.Sub(sample, _ops.Mul(modelOutput, sigma_hat));
            } else if (PredictionType == Prediction.V_Prediction) {
                predOriginalSample = _ops.Add(_ops.Mul(modelOutput, -sigma / MathF.Sqrt(MathF.Pow(sigma, 2f) + 1f)),
                    _ops.Div(sample, MathF.Pow(sigma, 2f) + 1f));
            } else {
                throw new ArgumentException($"Invalid prediction_type: {PredictionType}");
            }

            // 2. Convert to an ODE derivative
            TensorFloat derivative = _ops.Div(_ops.Sub(sample, predOriginalSample), sigma_hat);
            float dt = Sigmas[StepIndex.Value + 1] - sigma_hat;
            TensorFloat prev_sample = _ops.Add(sample, _ops.Mul(derivative, dt));

            // Increase step index by one
            StepIndex++;

            return new SchedulerOutput(prev_sample, predOriginalSample);
        }

        /// <inheritdoc/>
        protected override SchedulerOutput Step(TensorFloat modelOutput, float timestep, TensorFloat sample) {
            return Step(modelOutput, timestep, sample);
        }

        public override void Dispose() {
            AlphasCumprod?.Dispose();
            SigmasT?.Dispose();
            base.Dispose();
        }
    }
}