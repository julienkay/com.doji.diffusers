using static Doji.AI.Diffusers.ArrayUtils;
using System;
using Unity.Sentis;
using System.Linq;

namespace Doji.AI.Diffusers {

    public abstract class Scheduler : IDisposable {

        public SchedulerConfig Config { get; protected set; }

        /// <summary>
        /// standard deviation of the initial noise distribution
        /// </summary>
        public virtual float InitNoiseSigma { get { return 1; } }
        public virtual int Order { get { return 1; } }

        protected int NumTrainTimesteps { get => Config.NumTrainTimesteps; }
        protected float BetaStart { get => Config.BetaStart; }
        protected float BetaEnd { get => Config.BetaEnd; }
        protected Schedule BetaSchedule { get => Config.BetaSchedule; }
        protected bool SkipPrkSteps { get => Config.SkipPrkSteps; }
        protected bool SetAlphaToOne { get => Config.SetAlphaToOne; }
        protected int StepsOffset { get => Config.StepsOffset; }
        protected float[] TrainedBetas { get => Config.TrainedBetas; }
        protected Prediction PredictionType { get => Config.PredictionType; }
        protected Spacing TimestepSpacing { get => Config.TimestepSpacing; }
        protected bool RescaleBetasZeroSnr { get => Config.RescaleBetasZeroSnr; }
        
        protected Ops _ops;

        public Scheduler(BackendType backend) {
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        protected float[] GetBetas() {
            if (TrainedBetas != null) {
                return TrainedBetas;
            } else if (BetaSchedule == Schedule.Linear) {
                return Linspace(BetaStart, BetaEnd, NumTrainTimesteps);
            } else if (BetaSchedule == Schedule.ScaledLinear) {
                // this schedule is very specific to the latent diffusion model.
                return Linspace(MathF.Pow(BetaStart, 0.5f), MathF.Pow(BetaEnd, 0.5f), NumTrainTimesteps)
                    .Select(x => MathF.Pow(x, 2)).ToArray();
            } else if (BetaSchedule == Schedule.SquaredCosCapV2) {
                // Glide cosine schedule
                return BetasForAlphaBar(NumTrainTimesteps);
            } else {
                throw new NotImplementedException($"{BetaSchedule} is not implemented for {GetType().Name}");
            }
        }

        /// <summary>
        /// Create a beta schedule that discretizes the given alpha_t_bar function,
        /// which defines the cumulative product of (1-beta) over time from t = [0, 1].
        /// Contains a function alpha_bar that takes an argument t and transforms it to
        /// the cumulative product of(1-beta) up to that part of the diffusion process.
        /// </summary>
        /// <remarks>
        /// TODO: needs tests
        /// </remarks>
        private static float[] BetasForAlphaBar(
            int numDiffusionTimesteps,
            float maxBeta = 0.999f,
            AlphaTransform alphaTransformType = AlphaTransform.Cosine) {
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

        public virtual void Dispose() {
            _ops?.Dispose();
        }
    }
}