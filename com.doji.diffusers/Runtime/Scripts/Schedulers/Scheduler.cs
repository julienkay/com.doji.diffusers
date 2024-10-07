using static Doji.AI.ArrayUtils;
using System;
using Unity.Sentis;
using System.Linq;
using System.Collections.Generic;
using System.Collections;

namespace Doji.AI.Diffusers {

    public abstract class SchedulerFloat : Scheduler {
        public float[] Timesteps { get; protected set; }
        public override int TimestepsLength { get { return Timesteps.Length; } }
        protected SchedulerFloat(SchedulerConfig config, BackendType backend) : base(config, backend) { }
        public override IEnumerator<float> GetEnumerator() {
            foreach (float item in Timesteps) {
                yield return item;
            }
        }
        public override float[] GetTimestepsFromEnd(int n) {
            return Timesteps.Skip(TimestepsLength - n).ToArray();
        }
        public override float[] GetTimesteps() {
            return Timesteps;
        }
    }

    public abstract class SchedulerInt : Scheduler {
        public int[] Timesteps { get; protected set; }
        public override int TimestepsLength { get {  return Timesteps.Length; } }
        protected SchedulerInt(SchedulerConfig config, BackendType backend) : base(config, backend) { }
        public override IEnumerator<float> GetEnumerator() {
            foreach (float item in Timesteps) {
                yield return item;
            }
        }
        public override float[] GetTimestepsFromEnd(int n) {
            return Timesteps.Skip(TimestepsLength - n).Select(x => (float)x).ToArray();
        }
        public override float[] GetTimesteps() {
            float[] result = new float[Timesteps.Length];
            for (int i = 0; i < Timesteps.Length; i++) {
                result[i] = Timesteps[i];
            }
            return result;
        }
    }

    public abstract class Scheduler : IConfigurable<SchedulerConfig>, IDisposable, IEnumerable<float> {

        public SchedulerConfig Config { get; protected set; }

        public abstract int TimestepsLength { get; }
        protected Tensor<float> AlphasCumprod { get; set; }
        protected internal float[] AlphasCumprodF { get; set; }

        /// <summary>
        /// standard deviation of the initial noise distribution
        /// </summary>
        public    virtual float InitNoiseSigma    { get { return 1.0f; } }
        public    virtual int   Order             { get { return 1; } }
        public    int           NumInferenceSteps { get; protected set; }

        protected float         BetaStart                { get => Config.BetaStart.Value;                set => Config.BetaStart                = value; }
        protected float         BetaEnd                  { get => Config.BetaEnd.Value;                  set => Config.BetaEnd                  = value; }
        protected Schedule      BetaSchedule             { get => Config.BetaSchedule.Value;             set => Config.BetaSchedule             = value; }
        protected int           NumTrainTimesteps        { get => Config.NumTrainTimesteps.Value;        set => Config.NumTrainTimesteps        = value; }
        protected Prediction    PredictionType           { get => Config.PredictionType.Value;           set => Config.PredictionType           = value; }
        protected bool          SkipPrkSteps             { get => Config.SkipPrkSteps.Value;             set => Config.SkipPrkSteps             = value; }
        protected bool          SetAlphaToOne            { get => Config.SetAlphaToOne.Value;            set => Config.SetAlphaToOne            = value; }
        protected int           StepsOffset              { get => Config.StepsOffset.Value;              set => Config.StepsOffset              = value; }
        protected float[]       TrainedBetas             { get => Config.TrainedBetas;                   set => Config.TrainedBetas             = value; }
        protected internal      Spacing TimestepSpacing  { get => Config.TimestepSpacing.Value;          set => Config.TimestepSpacing          = value; }
        protected float         TimestepScaling          { get => Config.TimestepScaling.Value;          set => Config.TimestepScaling          = value; }
        protected bool          ClipSample               { get => Config.ClipSample.Value;               set => Config.ClipSample               = value; }
        protected float         ClipSampleRange          { get => Config.ClipSampleRange.Value;          set => Config.ClipSampleRange          = value; }
        protected bool          Thresholding             { get => Config.Thresholding.Value;             set => Config.Thresholding             = value; }
        protected float         DynamicThresholdingRatio { get => Config.DynamicThresholdingRatio.Value; set => Config.DynamicThresholdingRatio = value; }
        protected Interpolation InterpolationType        { get => Config.InterpolationType.Value;        set => Config.InterpolationType        = value; }
        protected float         SampleMaxValue           { get => Config.SampleMaxValue.Value;           set => Config.SampleMaxValue           = value; }
        protected bool          RescaleBetasZeroSnr      { get => Config.RescaleBetasZeroSnr.Value;      set => Config.RescaleBetasZeroSnr      = value; }
        protected Timestep      TimestepType             { get => Config.TimestepType.Value;             set => Config.TimestepType             = value; }
        protected int           OriginalInferenceSteps   { get => Config.OriginalInferenceSteps.Value;   set => Config.OriginalInferenceSteps   = value; }
        protected bool          UseKarrasSigmas          { get => Config.UseKarrasSigmas.Value;          set => Config.UseKarrasSigmas          = value; }
        protected float?        SigmaMin                 { get => Config.SigmaMin;                       set => Config.SigmaMin                 = value; }
        protected float?        SigmaMax                 { get => Config.SigmaMax;                       set => Config.SigmaMax                 = value; }


        /// <summary>
        /// Arguments passed into <see cref="Scheduler.Step(Tensor<float>, float, Tensor<float>)"/> method.
        /// </summary>
        public struct StepArgs {
            
            public Tensor<float> modelOutput;
            public float timestep;
            public Tensor<float> sample;
            public float eta;
            public bool useClippedModelOutput;
            public System.Random generator;
            public Tensor<float> varianceNoise;
            public float s_churn;
            public float s_tmin;
            public float s_tmax;
            public float s_noise;

            public StepArgs(Tensor<float> modelOutput,
                            float timestep,
                            Tensor<float> sample,
                            float eta = 0.0f,
                            bool useClippedModelOutput = false,
                            System.Random generator = null,
                            Tensor<float> varianceNoise = null,
                            float s_churn = 0.0f,
                            float s_tmin = 0.0f,
                            float s_tmax = float.PositiveInfinity,
                            float s_noise = 1.0f)
            {
                this.modelOutput = modelOutput;
                this.timestep = timestep;
                this.sample = sample;
                this.eta = eta;
                this.useClippedModelOutput = useClippedModelOutput;
                this.generator = generator;
                this.varianceNoise = varianceNoise;
                this.s_churn = s_churn;
                this.s_tmin = s_tmin;
                this.s_tmax = s_tmax;
                this.s_noise = s_noise;
            }
        }

        private StepArgs _args;

#pragma warning disable IDE1006 // Naming Styles
        /* StepArgs accessors for convenience */
        protected Tensor<float>   modelOutput           { get => _args.modelOutput; }
        protected float         timestep              { get => _args.timestep; }
        protected Tensor<float>   sample                { get => _args.sample;        set => _args.sample = value; }
        protected float         eta                   { get => _args.eta; }
        protected bool          useClippedModelOutput { get => _args.useClippedModelOutput; }
        protected System.Random generator             { get => _args.generator;     set => _args.generator = value; }
        protected Tensor<float>   varianceNoise         { get => _args.varianceNoise; set => _args.varianceNoise = value; }
        protected float s_churn { get => _args.s_churn; }
        protected float s_tmin  { get => _args.s_tmin; }
        protected float s_tmax  { get => _args.s_tmax; }
        protected float s_noise { get => _args.s_noise; }
#pragma warning restore IDE1006

        protected internal Ops _ops;

        public Scheduler(SchedulerConfig config, BackendType backend) {
            Config = config ?? new SchedulerConfig();
            _ops = new Ops(backend);
        }

        /// <summary>
        /// Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        /// </summary>
        public abstract void SetTimesteps(int numInferenceSteps);

        /// <summary>
        /// Predict the sample from the previous timestep by reversing the SDE.
        /// This function propagates the diffusion process from the learned model
        /// outputs (most often the predicted noise).
        /// </summary>
        public abstract SchedulerOutput Step(StepArgs args);

        /// <summary>
        /// Store args to allow for accessing them using properties for convenience.
        /// </summary>
        protected void SetStepArgs(StepArgs args) {
            _args = args;
        }

        public virtual Tensor<float> AddNoise(Tensor<float> originalSamples, Tensor<float> noise, Tensor<float> timesteps) {
            var alphasCumprod = _ops.GatherElements(AlphasCumprod, _ops.Cast(timesteps) as Tensor<int>, 0);
            var sqrtAlphaProd = _ops.Sqrt(alphasCumprod);
            while (sqrtAlphaProd.shape.rank < originalSamples.shape.rank) {
                sqrtAlphaProd.Reshape(sqrtAlphaProd.shape.Unsqueeze(-1)); // unsqueeze
            }

            var sqrtOneMinusAlphaProd = _ops.Sqrt(_ops.Sub(1.0f, alphasCumprod));
            while (sqrtOneMinusAlphaProd.shape.rank < originalSamples.shape.rank) {
                sqrtOneMinusAlphaProd.Reshape(sqrtOneMinusAlphaProd.shape.Unsqueeze(-1)); // unsqueeze
            }

            var tmp1 = _ops.Mul(sqrtAlphaProd, originalSamples);
            var tmp2 = _ops.Mul(sqrtOneMinusAlphaProd, noise);
            var noisySamples = _ops.Add(tmp1, tmp2);

            return noisySamples;
        }

        // TODO: just use '[^initTimestep..]' once all schedulers use float
        public abstract float[] GetTimestepsFromEnd(int n);
        public abstract float[] GetTimesteps();

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
        /// timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps)
        /// </summary>
        protected float[] GetTimeStepsLinspaceF() {
            int start = 0;
            int stop = NumTrainTimesteps - 1;
            int num = NumInferenceSteps;
            return ArrayUtils.Linspace(start, stop, num);
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

        protected float[] GetTimeStepsLeadingF() {
            int stepRatio = NumTrainTimesteps / NumInferenceSteps;
            int start = 0;
            int stop = NumInferenceSteps;
            int step = 1;

            int length = ((stop - start - 1) / step) + 1;
            float[] result = new float[length];

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
                result[length - i - 1] = (int)MathF.Round(value) - 1;
                value += step;
            }

            return result;
        }

        protected float[] GetTimeStepsTrailingF() {
            int start = NumTrainTimesteps;
            int stop = 0;
            float step = -NumTrainTimesteps / (float)NumInferenceSteps;

            int length = ((int)((stop - start - 1) / step)) + 1;
            float[] result = new float[length];

            float value = start;
            for (int i = 0; i < length; i++) {
                result[length - i - 1] = MathF.Round(value) - 1f;
                value += step;
            }

            return result;
        }

        /// <summary>
        /// Ensures interchangeability with schedulers that need to scale
        /// the denoising model input depending on the current timestep.
        /// </summary>
        public virtual Tensor<float> ScaleModelInput(Tensor<float> sample, float t) {
            return sample;
        }

        public virtual void Dispose() {
            _ops?.Dispose();
        }

        public abstract IEnumerator<float> GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }


        /// <summary>
        /// Instantiate a Scheduler from a JSON configuration file.
        /// </summary>
        public static Scheduler FromPretrained(DiffusionModel model, BackendType backend) {
            return IConfigurable<SchedulerConfig>.FromPretrained<Scheduler>(model.SchedulerConfig, backend);
        }

        /// <summary>
        /// Creates a scheduler of type <typeparamref name="T"/> with the given <paramref name="config"/>.
        /// This can be used to get a scheduler with the configuration of another one
        /// (in a way casting a certain type of scheduler to another one).
        /// </summary>
        protected static T FromConfig<T>(SchedulerConfig config, BackendType backend) where T : Scheduler {
            return (T)Activator.CreateInstance(typeof(T), config, backend);
        }
    }
}