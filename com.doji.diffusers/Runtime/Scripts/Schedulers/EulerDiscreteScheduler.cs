using static Doji.AI.Diffusers.ArrayUtils;
using System;
using Unity.Sentis;
using System.Collections.Generic;

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

        private int IndexForTimestep(float timestep, float[] scheduleTimesteps = null) {
            scheduleTimesteps ??= base.Timesteps as float[];

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

        /// <inheritdoc/>
        public override void SetTimesteps(int numInferenceSteps) {
            NumInferenceSteps = numInferenceSteps;

            // "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if (TimestepSpacing == Spacing.Linspace) {
                Timesteps = GetTimeStepsLinspaceF().Reverse();
            } else if (TimestepSpacing == Spacing.Leading) {
                Timesteps = GetTimeStepsLeadingF().Reverse();
            } else if (TimestepSpacing == Spacing.Trailing) {
                Timesteps = GetTimeStepsTrailingF().Reverse();
            } else {
                throw new ArgumentException($"{TimestepSpacing} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Spacing)))}.");
            }

            //float[] tmp1 = Sub(1f, _AlphasCumprod);
            //float[] tmp2 = tmp1.Div(_AlphasCumprod);
            //float[] sigmas = tmp2.Pow(0.5f);
            //float[] log_sigmas = sigmas.Log();
            var tmp1 = _ops.Sub(1f, AlphasCumprod);
            var tmp2 = _ops.Div(tmp1, AlphasCumprod);
            using TensorFloat pow = new TensorFloat(0.5f);
            var sigmas = _ops.Pow(tmp2, pow);
            var log_sigmas = _ops.Log(sigmas);

            if (InterpolationType == Interpolation.Linear) {
                sigmas = Interpolate(Timesteps, ArangeF(0, sigmas.Length), sigmas);
            } else  if (InterpolationType == Interpolation.LogLinear) {
                sigmas = Linspace(MathF.Log(sigmas[^1]), MathF.Log(sigmas[0]), NumInferenceSteps + 1).Exp();
            } else {
                throw new ArgumentException($"{InterpolationType} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Interpolation)))}.");
            }

            if (UseKarrasSigmas) {
                sigmas = ConvertToKarras(sigmas, NumInferenceSteps);
                using TensorFloat sigmasT = new TensorFloat(new TensorShape(sigmas.Length), sigmas);
                SigmaToT(sigmasT, log_sigmas)
            }

            /*
            if self.use_karras_sigmas:
                sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

            # TODO: Support the full EDM scalings for all prediction types and timestep types
            if self.config.timestep_type == "continuous" and self.config.prediction_type == "v_prediction":
                self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(device=device)
            else:
                self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
            self._step_index = None
            self._begin_index = None
            self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
             */
        }
        public float[] SigmaToT(float sigma, float[] logSigmas) {
            // get log sigma
            float logSigma = (float)Math.Log(Math.Max(sigma, 1e-10));

            float[] _dists = logSigmas > 0;
            // get distribution
            float[] dists = new float[logSigmas.Length];
            for (int i = 0; i < logSigmas.Length; i++) {
                dists[i] = logSigma - logSigmas[i];
            }

            // get sigmas range
            int[] lowIdx = new int[logSigmas.Length];
            for (int i = 0; i < logSigmas.Length; i++) {
                for (int j = 0; j < dists.Length; j++) {
                    if (dists[j] >= 0) {
                        lowIdx[i] = j;
                        break;
                    }
                }
            }
            int[] highIdx = new int[logSigmas.Length];
            for (int i = 0; i < lowIdx.Length; i++) {
                highIdx[i] = Math.Min(lowIdx[i] + 1, logSigmas.Length - 2);
            }

            float[] low = new float[lowIdx.Length];
            float[] high = new float[highIdx.Length];
            for (int i = 0; i < lowIdx.Length; i++) {
                low[i] = logSigmas[lowIdx[i]];
                high[i] = logSigmas[highIdx[i]];
            }

            // interpolate sigmas
            float[] w = new float[low.Length];
            for (int i = 0; i < low.Length; i++) {
                w[i] = (low[i] - logSigma) / (low[i] - high[i]);
                w[i] = Math.Min(Math.Max(w[i], 0), 1);
            }

            // transform interpolation to time range
            float[] t = new float[sigma.Length];
            for (int i = 0; i < sigma.Length; i++) {
                t[i] = (1 - w[i]) * lowIdx[i] + w[i] * highIdx[i];
            }

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
        protected override SchedulerOutput Step(TensorFloat modelOutput, float timestep, TensorFloat sample) {
            throw new NotImplementedException();
        }

        public override void Dispose() {
            AlphasCumprod?.Dispose();
            SigmasT?.Dispose();
            base.Dispose();
        }
    }
}