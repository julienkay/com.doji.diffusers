using static Doji.AI.ArrayUtils;
using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Ancestral sampling with Euler method steps.
    /// </summary>
    public class EulerAncestralDiscreteScheduler : EulerScheduler {

        public EulerAncestralDiscreteScheduler(SchedulerConfig config, BackendType backend = BackendType.GPUCompute) : base(config, backend) {
            Config.NumTrainTimesteps ??= 1000;
            Config.BetaStart ??= 0.0001f;
            Config.BetaEnd ??= 0.02f;
            Config.BetaSchedule ??= Schedule.Linear;
            Config.TrainedBetas ??= null;
            Config.PredictionType ??= Prediction.Epsilon;
            Config.TimestepSpacing ??= Spacing.Linspace;
            Config.StepsOffset ??= 0;
            Config.RescaleBetasZeroSnr ??= false;

            Initialize();
        }

        /// <inheritdoc/>
        public override void SetTimesteps(int numInferenceSteps) {
            base.SetTimesteps(numInferenceSteps);

            Sigmas = Interpolate(Timesteps, ArangeF(0, Sigmas.Length), Sigmas);
            Sigmas = Sigmas.Concatenate(0);
            StepIndex = null;
            BeginIndex = null;
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

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample;
            if (PredictionType == Prediction.Epsilon) {
                predOriginalSample = Ops.Sub(sample, Ops.Mul(modelOutput, sigma));
            } else if (PredictionType == Prediction.V_Prediction) {
                // * c_out + input * c_skip
                predOriginalSample = Ops.Add(Ops.Mul(modelOutput, -sigma / MathF.Sqrt(MathF.Pow(sigma, 2f) + 1f)),
                    Ops.Div(sample, MathF.Pow(sigma, 2f) + 1f));
            } else if (PredictionType == Prediction.Sample) {
                throw new NotImplementedException("'PredictionType' not implemented yet: sample");
            } else {
                throw new ArgumentException($"Invalid PredictionType: {PredictionType}");
            }

            float sigmaFrom = Sigmas[StepIndex.Value];
            float sigmaTo = Sigmas[StepIndex.Value + 1];
            float sigmaUp = MathF.Sqrt(MathF.Pow(sigmaTo, 2f) * (MathF.Pow(sigmaFrom, 2f) - MathF.Pow(sigmaTo, 2f)) / MathF.Pow(sigmaFrom, 2f));
            float sigmaDown = MathF.Sqrt(MathF.Pow(sigmaTo, 2f) - MathF.Pow(sigmaUp, 2f));

            // 2. Convert to an ODE derivative
            Tensor<float> derivative = Ops.Div(Ops.Sub(sample, predOriginalSample), sigma);
            float dt = sigmaDown - sigma;
            Tensor<float> prevSample = Ops.Add(sample, Ops.Mul(derivative, dt));

            int seed = generator.Next();
            var noise = Ops.RandomNormal(modelOutput.shape, 0, 1, seed);

            prevSample = Ops.Add(prevSample, Ops.Mul(noise, sigmaUp));

            // upon completion increase step index by one
            StepIndex++;

            return new SchedulerOutput(prevSample, predOriginalSample);
        }

        public static EulerAncestralDiscreteScheduler FromConfig(SchedulerConfig cfg, BackendType b) => FromConfig<EulerAncestralDiscreteScheduler>(cfg, b);
    }
}