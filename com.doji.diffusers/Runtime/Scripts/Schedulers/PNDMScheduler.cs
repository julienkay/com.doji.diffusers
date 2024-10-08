﻿using System.Collections.Generic;
using System.Linq;
using System;
using static Doji.AI.ArrayUtils;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public enum AlphaTransform { Cosine, Exp }

    public class PNDMScheduler : SchedulerInt {

        public Tensor<float> Betas { get; private set; }
        public Tensor<float> Alphas { get; private set; }
        public float FinalAlphaCumprod { get; private set; }

        public int PndmOrder { get; private set; }
        public Tensor<float> CurModelOutput { get; set; }
        public int Counter { get; set; }
        public Tensor<float> CurSample { get; set; }
        public List<Tensor<float>> Ets { get; private set; }
        public int[] PrkTimesteps { get; set; }
        public int[] PlmsTimesteps { get; set; }
        public bool AcceptsEta { get { return false; } }

        public PNDMScheduler(SchedulerConfig config = null, BackendType backend = BackendType.GPUCompute) : base(config, backend) {
            Config.NumTrainTimesteps ??= 1000;
            Config.BetaStart         ??= 0.0001f;
            Config.BetaEnd           ??= 0.02f;
            Config.BetaSchedule      ??= Schedule.Linear;
            Config.TrainedBetas      ??= null;
            Config.SkipPrkSteps      ??= false;
            Config.SetAlphaToOne     ??= false;
            Config.PredictionType    ??= Prediction.Epsilon;
            Config.TimestepSpacing   ??= Spacing.Leading;
            Config.StepsOffset       ??= 0;

            Ets = new List<Tensor<float>>();

            float[] betas = GetBetas();
            Betas = new Tensor<float>(new TensorShape(betas.Length), betas);
            Alphas = Ops.Sub(1.0f, Betas);
            float[] alphas = betas.Select(beta => 1.0f - beta).ToArray();
            AlphasCumprodF = alphas.CumProd();
            AlphasCumprod = new Tensor<float>(new TensorShape(alphas.Length), AlphasCumprodF);
            FinalAlphaCumprod = SetAlphaToOne ? 1.0f : AlphasCumprodF[0];

            // For now we only support F-PNDM, i.e. the runge-kutta method
            // For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
            // mainly at formula (9), (12), (13) and the Algorithm 2.
            PndmOrder = 4;
            NumInferenceSteps = 0;
            Timesteps = Arange(0, NumTrainTimesteps).Reverse();
        }

        /// <inheritdoc/>
        public override void SetTimesteps(int numInferenceSteps) {
            NumInferenceSteps = numInferenceSteps;

            // "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if (TimestepSpacing == Spacing.Linspace) {
                Timesteps = GetTimeStepsLinspace();
            } else if (TimestepSpacing == Spacing.Leading) {
                Timesteps = GetTimeStepsLeading();
            } else if (TimestepSpacing == Spacing.Trailing) {
                Timesteps = GetTimeStepsTrailing();
            } else {
                throw new ArgumentException($"{TimestepSpacing} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Spacing)))}.");
            }
            
            if (SkipPrkSteps) {
                PrkTimesteps = new int[0];
                // for some models like stable diffusion the prk steps can/should be skipped to
                // produce better results. When using PNDM with SkipPrkSteps the implementation
                // is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
                PlmsTimesteps = Concatenate(
                    Timesteps[..^1],
                    Timesteps[^2..^1],
                    Timesteps[^1..])
                    .Reverse();
                Timesteps = PlmsTimesteps;
            } else {
                int[] tileArray = new int[] { 0, NumTrainTimesteps / NumInferenceSteps / 2 }.Tile(PndmOrder);
                PrkTimesteps = Timesteps[^PndmOrder..].Repeat(2).Add(tileArray);
                PrkTimesteps = PrkTimesteps[..^1].Repeat(2)[1..^1].Reverse();
                PlmsTimesteps = Timesteps[..^3].Reverse();

                Timesteps = PrkTimesteps.Concatenate(PlmsTimesteps);
            }

            ClearEts();
            Counter = 0;
            CurModelOutput = null;
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Calls <see cref="StepPrk(Tensor<float>, int, Tensor<float>)"/> or
        /// <see cref="StepPlms(Tensor<float>, int, Tensor<float>)"/> depending
        /// on the internal variable <see cref="Counter"/>.
        /// </remarks>
        public override SchedulerOutput Step(StepArgs args) {
            SetStepArgs(args);
            if (Counter < PrkTimesteps.Length && !SkipPrkSteps) {
                return StepPrk(modelOutput, (int)timestep, sample);
            } else {
                return StepPlms(modelOutput, (int)timestep, sample);
            }
        }

        /// <summary>
        /// Predict the sample from the previous timestep by reversing the SDE.
        /// This function propagates the sample with the Runge-Kutta method.
        /// It performs four forward passes to approximate the solution to the
        /// differential equation.
        /// </summary>
        private SchedulerOutput StepPrk(Tensor<float> modelOutput, int timestep, Tensor<float> sample) {
            if (NumInferenceSteps == 0) {
                throw new ArgumentException("Number of inference steps is '0', you need to run 'SetTimesteps' after creating the scheduler");
            }

            int diffToPrev = (Counter % 2 != 0) ? 0 : NumTrainTimesteps / NumInferenceSteps / 2;
            int prevTimestep = timestep - diffToPrev;
            timestep = PrkTimesteps[Counter / 4 * 4];

            if (CurModelOutput == null) {
                using var init = new Tensor<float>(modelOutput.shape);
                CurModelOutput = init;
            }

            if (Counter % 4 == 0) {
                var tmp = Ops.Div(modelOutput, 6f);
                CurModelOutput = Ops.Add(CurModelOutput, tmp);
                modelOutput = TakeOwnership(modelOutput);
                Ets.Add(modelOutput);
                CurSample = sample;
            } else if ((Counter - 1) % 4 == 0) {
                var tmp = Ops.Div(modelOutput, 3f);
                CurModelOutput = Ops.Add(CurModelOutput, tmp);
            } else if ((Counter - 2) % 4 == 0) {
                var tmp = Ops.Div(modelOutput, 3f);
                CurModelOutput = Ops.Add(CurModelOutput, tmp);
            } else if ((Counter - 3) % 4 == 0) {
                var tmp = Ops.Div(modelOutput, 6f);
                modelOutput = Ops.Add(CurModelOutput, tmp);
                CurModelOutput = null;
            }

            // CurSample should not be `null`
            Tensor<float> curSample = (CurSample != null) ? CurSample : sample;

            Tensor<float> prevSample = GetPrevSample(curSample, timestep, prevTimestep, modelOutput);
            Counter++;

            return new SchedulerOutput(prevSample);
        }

        /// <summary>
        /// Predict the sample from the previous timestep by reversing the SDE.
        /// This function propagates the sample with the linear multistep method.
        /// It performs one forward pass multiple times to approximate the solution.
        /// </summary>
        private SchedulerOutput StepPlms(Tensor<float> modelOutput, int timestep, Tensor<float> sample) {
            if (NumInferenceSteps == 0) {
                throw new ArgumentException("Number of inference steps is 0. " +
                    "Make sure to run SetTimesteps() after creating the scheduler");
            }

            if (!SkipPrkSteps && Ets.Count < 3) {
                throw new ArgumentException($"{GetType()} can only be run AFTER scheduler has been run " +
                    "in 'prk' mode for at least 12 iterations.");
            }

            int prevTimestep = timestep - NumTrainTimesteps / NumInferenceSteps;

            if (Counter != 1) {
                // only keep last 3
                for (int i = (Ets.Count - 3) - 1; i >= 0; i--) {
                    Ets[i].Dispose();
                    Ets.RemoveAt(i);
                }
                modelOutput = TakeOwnership(modelOutput);
                Ets.Add(modelOutput);
            } else {
                prevTimestep = timestep;
                timestep += NumTrainTimesteps / NumInferenceSteps;
            }

            if (Ets.Count == 1 && Counter == 0) {
                CurSample = sample;
            } else if (Ets.Count == 1 && Counter == 1) {
                var tmp = Ops.Add(modelOutput, Ets[^1]);
                modelOutput = Ops.Div(tmp, 2f);
                sample = CurSample;
                CurSample = null;
            } else if (Ets.Count == 2) {
                var tmp = Ops.Mul(3f, Ets[^1]);
                var tmp2 = Ops.Sub(tmp, Ets[^2]);
                modelOutput = Ops.Div(tmp2, 2f);
            } else if (Ets.Count == 3) {
                var tmp = Ops.Mul(23f, Ets[^1]);
                var tmp2 = Ops.Mul(16f, Ets[^2]);
                var tmp3 = Ops.Sub(tmp, tmp2);
                var tmp4 = Ops.Mul(5f, Ets[^3]);
                var tmp5 = Ops.Add(tmp3, tmp4);
                modelOutput = Ops.Div(tmp5, 12f);
            } else {
                var tmp = Ops.Mul(55f, Ets[^1]);
                var tmp2 = Ops.Mul(59f, Ets[^2]);
                var tmp3 = Ops.Sub(tmp, tmp2);
                var tmp4 = Ops.Mul(37f, Ets[^3]);
                var tmp5 = Ops.Add(tmp3, tmp4);
                var tmp6 = Ops.Mul(9f, Ets[^4]);
                var tmp7 = Ops.Sub(tmp5, tmp6);
                modelOutput = Ops.Div(tmp7, 24f);
            }

            Tensor<float> prevSample = GetPrevSample(sample, timestep, prevTimestep, modelOutput);
            Counter++;

            return new SchedulerOutput(prevSample);
        }

        /// <summary>
        /// See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        /// this function computes x_(t−δ) using the formula of (9)
        /// Note that x_t needs to be added to both sides of the equation
        /// </summary>
        private Tensor<float> GetPrevSample(Tensor<float> sample, int timestep, int prevTimestep, Tensor<float> modelOutput) {
            // Notation (<variable name> -> <name in paper>
            // alphaProdT -> α_t
            // alphaProdTPrev -> α_(t−δ)
            // betaProdT -> (1 - α_t)
            // betaProdTPrev -> (1 - α_(t−δ))
            // sample -> x_t
            // model_output -> e_θ(x_t, t)
            // prev_sample -> x_(t−δ)

            double alphaProdT = AlphasCumprodF[timestep];
            double alphaProdTPrev = (prevTimestep >= 0) ? AlphasCumprodF[prevTimestep] : FinalAlphaCumprod;
            double betaProdT = 1.0 - alphaProdT;
            double betaProdTPrev = 1.0 - alphaProdTPrev;

            if (PredictionType == Prediction.V_Prediction) {
                var a = Ops.Mul((float)Math.Sqrt(alphaProdT), modelOutput);
                var b = Ops.Mul((float)Math.Sqrt(betaProdT), sample);
                modelOutput = Ops.Add(a, b);
            } else if (PredictionType != Prediction.Epsilon) {
                throw new ArgumentException($"prediction_type given as {PredictionType} must be one of `epsilon` or `v_prediction`");
            }

            // corresponds to (α_(t−δ) - α_t) divided by
            // denominator of x_t in formula (9) and plus 1
            // Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
            // sqrt(α_(t−δ)) / sqrt(α_t))
            double sampleCoeff = Math.Pow(alphaProdTPrev / alphaProdT, 0.5);

            // corresponds to denominator of e_θ(x_t, t) in formula (9)
            double modelOutputDenomCoeff = alphaProdT * Math.Pow(betaProdTPrev, 0.5) +
                Math.Pow(alphaProdT * betaProdT * alphaProdTPrev, 0.5);

            // full formula (9)
            var tmp = Ops.Mul((float)sampleCoeff, sample);
            var tmp2 = Ops.Mul((float)(alphaProdTPrev - alphaProdT), modelOutput);
            var tmp3 = Ops.Div(tmp2, (float)modelOutputDenomCoeff);
            var prevSample = Ops.Sub(tmp, tmp3);

            return prevSample;
        }

        private void ClearEts() {
            foreach(Tensor t in Ets) {
                t.Dispose();
            }
            Ets?.Clear();
        }

        private Tensor<float> TakeOwnership(Tensor<float> X) {
            Tensor<float> O = Ops.Copy(X);
            Ops.TakeOwnership(O);
            return O;
        }

        public override void Dispose() {
            ClearEts();
            Betas?.Dispose();
            AlphasCumprod?.Dispose();
            base.Dispose();
        }

        public static PNDMScheduler FromConfig(SchedulerConfig cfg, BackendType b) => FromConfig<PNDMScheduler>(cfg, b);
    }
}