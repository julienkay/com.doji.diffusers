using static Doji.AI.Diffusers.ArrayUtils;
using System;
using Unity.Sentis;
using System.Collections.Generic;

namespace Doji.AI.Diffusers {

    public class EulerDiscreteScheduler : SchedulerFloat {

        public float[] Betas { get; private set; }
        public TensorFloat AlphasCumprod { get; private set; }
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
            float[] alphasCumprod = alphas.CumProd();
            AlphasCumprod = new TensorFloat(new TensorShape(alphas.Length), alphasCumprod);

            if (RescaleBetasZeroSnr) {
                // Close to 0 without being 0 so first sigma is not inf
                // FP16 smallest positive subnormal works well here
                //AlphasCumprod[^1] = MathF.Pow(2f, -24f);
                throw new NotImplementedException();
            }

            var tmp1 = Sub(1f, alphasCumprod);
            var tmp2 = tmp1.Div(alphasCumprod);
            var sigmas = tmp2.Pow(0.5f).Reverse();
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
            throw new NotImplementedException();
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