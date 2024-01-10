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
        public float InitNoiseSigma { get; private set; }
        public int PndmOrder { get; private set; }
        public int CurModelOutput { get; set; }
        public int Counter { get; set; }
        public object CurSample { get; set; }
        public List<object> Ets { get; private set; }
        public int NumInferenceSteps { get; set; }
        public int[] Timesteps { get; private set; }
        public int[] PrkTimesteps { get; set; }
        public int[] PlmsTimesteps { get; set; }

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
            Counter = 0;
            CurSample = null;

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

            InitNoiseSigma = 1.0f;

            // For now we only support F-PNDM, i.e. the runge-kutta method
            // For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
            // mainly at formula (9), (12), (13) and the Algorithm 2.
            PndmOrder = 4;
            CurModelOutput = 0;
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
                // # for some models like stable diffusion the prk steps can/should be skipped to
                // # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
                // # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
                PlmsTimesteps = Timesteps[..^1].Concat(Timesteps[^2..^1])
                                               .Concat(Timesteps[^1..])
                                               .Reverse()
                                               .ToArray();
                Timesteps = PlmsTimesteps;
            } else {
                //throw new NotImplementedException("SkipPrkSteps not implemented yet.");
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