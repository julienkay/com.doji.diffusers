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
        public int NumInferenceSteps { get; protected set; }
        public int[] Timesteps { get; protected set; }


        protected float BetaStart { get => Config.BetaStart; }
        protected float BetaEnd { get => Config.BetaEnd; }
        protected Schedule BetaSchedule { get => Config.BetaSchedule; }
        protected bool ClipSample { get => Config.ClipSample; }
        protected float ClipSampleRange { get => Config.ClipSampleRange; }
        protected float DynamicThresholdingRatio { get => Config.DynamicThresholdingRatio; }
        protected int NumTrainTimesteps { get => Config.NumTrainTimesteps; }
        protected Prediction PredictionType { get => Config.PredictionType; }
        protected float SampleMaxValue { get => Config.SampleMaxValue; }
        protected bool SkipPrkSteps { get => Config.SkipPrkSteps; }
        protected bool SetAlphaToOne { get => Config.SetAlphaToOne; }
        protected int StepsOffset { get => Config.StepsOffset; }
        protected float[] TrainedBetas { get => Config.TrainedBetas; }
        protected bool Thresholding { get => Config.Thresholding; }
        protected internal Spacing TimestepSpacing { get => Config.TimestepSpacing; }
        protected bool RescaleBetasZeroSnr { get => Config.RescaleBetasZeroSnr; }
        
        protected Ops _ops;

        public Scheduler(BackendType backend) {
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        /// <summary>
        /// Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        /// </summary>
        public abstract void SetTimesteps(int numInferenceSteps);

        /// <summary>
        /// Predict the sample from the previous timestep by reversing the SDE.
        /// This function propagates the diffusion process from the learned model
        /// outputs (most often the predicted noise), and calls step_prk or
        /// step_plms depending on the internal variable <see cref="Counter"/>.
        /// </summary>
        protected abstract SchedulerOutput Step(TensorFloat modelOutput, int timestep, TensorFloat sample);

        /// <inheritdoc cref="Step"/>
        /// <remarks>
        /// Override this method only in DDIMSCheduler which takes an additional eta parameter.
        /// </remarks>
        public virtual SchedulerOutput Step(
            TensorFloat modelOutput,
            int timestep,
            TensorFloat sample,
            float eta = 0.0f,
            bool useClippedModelOutput = false,
            System.Random generator = null,
            TensorFloat varianceNoise = null)
        {
            return Step(modelOutput, timestep, sample);
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

        /// <summary>
        /// timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps).round()
        /// </summary>
        protected int[] GetTimeStepsLinspace() {
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
        protected int[] GetTimeStepsLeading() {
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
        protected int[] GetTimeStepsTrailing() {
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

        /// <summary>
        /// Ensures interchangeability with schedulers that need to scale
        /// the denoising model input depending on the current timestep.
        /// </summary>
        public TensorFloat ScaleModelInput(TensorFloat latentModelInput, int t) {
            return latentModelInput;
        }

        public virtual void Dispose() {
            _ops?.Dispose();
        }
    }
}