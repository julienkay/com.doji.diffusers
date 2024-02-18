using static Doji.AI.Diffusers.ArrayUtils;
using System;
using Unity.Sentis;
using Unity.Sentis.Layers;
using System.Collections.Generic;

namespace Doji.AI.Diffusers {

    public class EulerDiscreteScheduler : Scheduler {

        public float[] Betas { get; private set; }
        public TensorFloat AlphasCumprod { get; private set; }
        public TensorFloat SigmasT { get; protected set; }
        private float[] Sigmas { get; set; }
        public float[] TimestepsF { get; private set; }

        public bool IsScaleInputCalled { get; private set; }

        private int _stepIndex;
        private int _beginIndex;

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
            TimestepsF = Linspace(0f, NumTrainTimesteps - 1, NumTrainTimesteps).Reverse();

            // setable values
            NumInferenceSteps = 0;

            // TODO: Support the full EDM scalings for all prediction types and timestep types
            if (TimestepType == Timestep.Continuous && PredictionType == Prediction.V_Prediction) {
                for (int i = 0; i < sigmas.Length; i++) {
                    TimestepsF[i] = 0.25f * MathF.Log(sigmas[i]);
                }
            }

            sigmas = sigmas.Concatenate(0);

            IsScaleInputCalled = false;
            _stepIndex = 0;
            _beginIndex = 0;
            SigmasT = new TensorFloat(new TensorShape(sigmas.Length), sigmas);
        }


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

        public override void SetTimesteps(int numInferenceSteps) {
            throw new NotImplementedException();
        }

        protected override SchedulerOutput Step(TensorFloat modelOutput, int timestep, TensorFloat sample) {
            throw new NotImplementedException();
        }

        public override void Dispose() {
            AlphasCumprod?.Dispose();
            SigmasT?.Dispose();
            base.Dispose();
        }
    }
}