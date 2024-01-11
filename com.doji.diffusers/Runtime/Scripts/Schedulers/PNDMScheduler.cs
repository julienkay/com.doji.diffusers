using System.Collections.Generic;
using System.Linq;
using System;

namespace Doji.AI.Diffusers {

    public enum Schedule { Linear, ScaledLinear, SquaredCosCapV2 }

    public enum Prediction {

        /// <summary>
        /// predicts the noise of the diffusion process
        /// </summary>
        Epsilon,

        /// <summary>
        /// directly predicts the noisy sample
        /// </summary>
        Sample,

        /// <summary>
        /// see section 2.4 of Imagen Video paper
        /// </summary>
        V_Prediction
    }

    public enum Spacing { Leading, Trailing, Linspace }

    public enum AlphaTransform { Cosine, Exp }

    public class PNDMScheduler {

        public int Order { get { return 1; } }
        public int NumTrainTimesteps { get; set; }
        public float BetaStart { get; set; }
        public float BetaEnd { get; set; }
        public Schedule BetaSchedule { get; set; }
        public bool SkipPrkSteps { get; set; }
        public Prediction PredictionType { get; set; }
        public Spacing TimestepSpacing { get; set; }
        public int StepsOffset { get; set; }
        public float[] Betas { get; private set; }
        public float[] Alphas { get; private set; }
        public float[] AlphasCumprod { get; private set; }
        public float FinalAlphaCumprod { get; private set; }
        public float InitNoiseSigma { get { return 1.0f; } }
        public int PndmOrder { get; private set; }
        public int CurModelOutput { get; set; }
        public int Counter { get; set; }
        public object CurSample { get; set; }
        public List<object> Ets { get; private set; }
        public int NumInferenceSteps { get; set; }
        public int[] Timesteps { get; private set; }
        public int[] PrkTimesteps { get; set; }
        public int[] PlmsTimesteps { get; set; }
        public bool AcceptsEta { get { return false; } }

        public PNDMScheduler(
            int numTrainTimesteps = 1000,
            float betaStart = 0.0001f,
            float betaEnd = 0.02f,
            Schedule betaSchedule = Schedule.Linear,
            float[] trainedBetas = null,
            bool skipPrkSteps = false,
            bool setAlphaToOne = false,
            Prediction predictionType = Prediction.Epsilon,
            Spacing timestepSpacing = Spacing.Leading,
            int stepsOffset = 0)
        {
            NumTrainTimesteps = numTrainTimesteps;
            BetaStart = betaStart;
            BetaEnd = betaEnd;
            BetaSchedule = betaSchedule;
            SkipPrkSteps = skipPrkSteps;
            PredictionType = predictionType;
            TimestepSpacing = timestepSpacing;
            StepsOffset = stepsOffset;
            Ets = new List<object>();

            if (trainedBetas != null) {
                Betas = trainedBetas;
            } else if (betaSchedule == Schedule.Linear) {
                Betas = Enumerable.Range(0, numTrainTimesteps)
                                  .Select(i => betaStart + (betaEnd - betaStart) * i / (numTrainTimesteps - 1))
                                  .ToArray();
            } else if (betaSchedule == Schedule.ScaledLinear) {
                Betas = Enumerable.Range(0, numTrainTimesteps)
                                  .Select(i => MathF.Pow(betaStart, 0.5f) + (MathF.Pow(betaEnd, 0.5f) - MathF.Pow(betaStart, 0.5f)) * i / (numTrainTimesteps - 1))
                                  .Select(x => MathF.Pow(x, 2))
                                  .ToArray();
            } else if (betaSchedule == Schedule.SquaredCosCapV2) {
                Betas = BetasForAlphaBar(numTrainTimesteps);
            } else {
                throw new NotImplementedException($"{betaSchedule} is not implemented for {GetType().Name}");
            }

            Alphas = Betas.Select(beta => 1.0f - beta).ToArray();
            AlphasCumprod = Alphas.CumProd();
            FinalAlphaCumprod = setAlphaToOne ? 1.0f : AlphasCumprod[0];

            // For now we only support F-PNDM, i.e. the runge-kutta method
            // For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
            // mainly at formula (9), (12), (13) and the Algorithm 2.
            PndmOrder = 4;
            Timesteps = Enumerable.Range(0, numTrainTimesteps).Reverse().ToArray();
        }

        /// <summary>
        /// Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        /// </summary>
        public void SetTimesteps(int numInferenceSteps) {
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
                PlmsTimesteps = Timesteps[..^1].Concat(Timesteps[^2..^1])
                                               .Concat(Timesteps[^1..])
                                               .Reverse()
                                               .ToArray();
                Timesteps = PlmsTimesteps;
            } else {
                throw new NotImplementedException("PrkTimesteps not implemented yet.");
            }

            Ets = new List<object>();
            Counter = 0;
            CurModelOutput = 0;
        }

        private static float[] BetasForAlphaBar(
            int numDiffusionTimesteps,
            float maxBeta = 0.999f,
            AlphaTransform alphaTransformType = AlphaTransform.Cosine)
        {
            float[] betas = new float[numDiffusionTimesteps];

            Func<float, float> alphaBarFn;
            if (alphaTransformType == AlphaTransform.Cosine) {
                alphaBarFn = t => (float)Math.Pow(Math.Cos((t + 0.008) / 1.008 * Math.PI / 2), 2);
            } else if (alphaTransformType == AlphaTransform.Exp) {
                alphaBarFn = t => (float)Math.Exp(t * -12.0);
            } else {
                throw new ArgumentException($"Unsupported alpha_transform_type: {alphaTransformType}");
            }

            for (int i = 0; i < numDiffusionTimesteps; i++) {
                float t1 = i / (float)numDiffusionTimesteps;
                float t2 = (i + 1) / (float)numDiffusionTimesteps;
                betas[i] = Math.Min(1 - alphaBarFn(t2) / alphaBarFn(t1), maxBeta);
            }

            return betas;
        }

        /// <summary>
        /// Predict the sample from the previous timestep by reversing the SDE.
        /// This function propagates the diffusion process from the learned model
        /// outputs (most often the predicted noise), and calls step_prk or
        /// step_plms depending on the internal variable <see cref="Counter"/>.
        /// </summary>
        public SchedulerOutput Step(float[] modelOutput, int timestep, float[] sample) {
            if (Counter < PrkTimesteps.Length && !SkipPrkSteps) {
                return StepPrk(modelOutput: modelOutput, timestep: timestep, sample: sample);
            } else {
                return StepPlms(modelOutput: modelOutput, timestep: timestep, sample: sample);
            }
        }

        /// <summary>
        /// Predict the sample from the previous timestep by reversing the SDE.
        /// This function propagates the sample with the Runge-Kutta method.
        /// It performs four forward passes to approximate the solution to the
        /// differential equation.
        /// </summary>
        private SchedulerOutput StepPrk(float[] modelOutput, int timestep, float[] sample) {
            throw new NotImplementedException("PrkTimesteps not implemented yet.");
        }

        /// <summary>
        /// Predict the sample from the previous timestep by reversing the SDE.
        /// This function propagates the sample with the linear multistep method.
        /// It performs one forward pass multiple times to approximate the solution.
        /// </summary>
        private SchedulerOutput StepPlms(float[] modelOutput, int timestep, float[] sample) {
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
                Ets = Ets.GetRange(Math.Max(0, Ets.Count - 3), Math.Min(Ets.Count, 3));
                Ets.Add(modelOutput);
            } else {
                prevTimestep = timestep;
                timestep += NumTrainTimesteps / NumInferenceSteps;
            }

            float[] prevSample = GetPrevSample(sample, timestep, prevTimestep, modelOutput);
            Counter++;

            return new SchedulerOutput(prevSample);
        }

        /// <summary>
        /// Ensures interchangeability with schedulers that need to scale
        /// the denoising model input depending on the current timestep.
        /// </summary>
        public float[] ScaleModelInput(float[] latentModelInput, int t) {
            return latentModelInput;
        }

        /// <summary>
        /// See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        /// this function computes x_(t−δ) using the formula of (9)
        /// Note that x_t needs to be added to both sides of the equation
        /// </summary>
        private float[] GetPrevSample(float[] sample, int timestep, int prevTimestep, float[] modelOutput) {
            // Notation (<variable name> -> <name in paper>
            // alphaProdT -> α_t
            // alphaProdTPrev -> α_(t−δ)
            // betaProdT -> (1 - α_t)
            // betaProdTPrev -> (1 - α_(t−δ))
            // sample -> x_t
            // model_output -> e_θ(x_t, t)
            // prev_sample -> x_(t−δ)

            double alphaProdT = AlphasCumprod[timestep];
            double alphaProdTPrev = (prevTimestep >= 0) ? AlphasCumprod[prevTimestep] : FinalAlphaCumprod;
            double betaProdT = 1 - alphaProdT;
            double betaProdTPrev = 1 - alphaProdTPrev;

            if (PredictionType == Prediction.V_Prediction) {
                for (int i = 0; i < modelOutput.Length; i++) {
                    modelOutput[i] = (float)(Math.Sqrt(alphaProdT) * modelOutput[i] + Math.Sqrt(betaProdT) * sample[i]);
                }
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
            float[] prevSample = new float[sample.Length];
            for (int i = 0; i < sample.Length; i++) {
                prevSample[i] = (float)(sampleCoeff * sample[i] - (alphaProdTPrev - alphaProdT) * modelOutput[i] / modelOutputDenomCoeff);
            }

            return prevSample;
        }

        /// <summary>
        /// timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps).round()
        /// </summary>
        private int[] GetTimeStepsLinspace() {
            int start = 0;
            int stop = NumTrainTimesteps - 1;
            int num = NumInferenceSteps;

            int[] result = new int[num];
            float step = (stop - start) / (float)(num - 1);

            for (int i = 0; i < num; i++) {
                result[i] = (int)Math.Round(start + i * step);
            }

            return result;
        }

        /// <summary>
        /// timesteps = np.arange(0, num_inference_steps) * step_ratio).round()
        /// timesteps += steps_offset
        /// </summary>
        private int[] GetTimeStepsLeading() {
            int stepRatio = NumTrainTimesteps / NumInferenceSteps;
            int start = 0;
            int stop = NumInferenceSteps;
            int step = 1;

            int length = ((stop - start - 1) / step) + 1;
            int[] result = new int[length];

            for (int i = 0, value = start; i < length; i++, value += step) {
                result[i] = (value * stepRatio) + StepsOffset;
            }

            return result;
        }

        /// <summary>
        /// timesteps = np.round(np.arange(num_train_timesteps, 0, -step_ratio))[::-1]
        /// timesteps -= 1
        /// </summary>
        private int[] GetTimeStepsTrailing() {
            int start = NumTrainTimesteps;
            int stop = 0;
            float step = -NumTrainTimesteps / (float)NumInferenceSteps;

            int length = ((int)((stop - start - 1) / step)) + 1;
            int[] result = new int[length];

            float value = start;
            for (int i = 0; i < length; i++) {
                result[length - i - 1] = (int)Math.Round(value) - 1;
                value += step;
            }

            return result;
        }
    }
}