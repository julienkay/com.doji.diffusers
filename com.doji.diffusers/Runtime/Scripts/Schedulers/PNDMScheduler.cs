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


    public class PNDMScheduler {

        public int NumTrainTimesteps { get; set; }
        public float BetaStart { get; set; }
        public float BetaEnd { get; set; }
        public Schedule BetaSchedule { get; set; }
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
        public int? NumInferenceSteps { get; set; }
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
            AlphasCumprod = CumProd(Alphas);
            FinalAlphaCumprod = setAlphaToOne ? 1.0f : AlphasCumprod[0];

            InitNoiseSigma = 1.0f;

            PndmOrder = 4;
            CurModelOutput = 0;
            NumInferenceSteps = null;
            Timesteps = Enumerable.Range(0, numTrainTimesteps).Reverse().ToArray();
            PrkTimesteps = null;
            PlmsTimesteps = null;
        }

        private static float[] BetasForAlphaBar(int numDiffusionTimesteps, float maxBeta = 0.999f, string alphaTransformType = "cosine") {
            float[] betas = new float[numDiffusionTimesteps];

            Func<float, float> alphaBarFn;
            if (alphaTransformType == "cosine") {
                alphaBarFn = t => (float)Math.Pow(Math.Cos((t + 0.008) / 1.008 * Math.PI / 2), 2);
            } else if (alphaTransformType == "exp") {
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

        private static float[] CumProd(float[] array) {
            int length = array.Length;
            float[] result = new float[length];
            float product = 1.0f;

            for (int i = 0; i < length; i++) {
                product *= array[i];
                result[i] = product;
            }

            return result;
        }
    }
}