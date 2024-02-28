using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model.
    /// </summary>
    public enum Schedule {
        [EnumMember(Value = "linear")]
        Linear,

        [EnumMember(Value = "scaled_linear")]
        ScaledLinear,

        [EnumMember(Value = "squaredcos_cap_v2")]
        SquaredCosCapV2
    }

    /// <summary>
    /// Prediction type of the scheduler function.
    /// </summary>
    public enum Prediction {

        /// <summary>
        /// predicts the noise of the diffusion process
        /// </summary>
        [EnumMember(Value = "epsilon")]
        Epsilon,

        /// <summary>
        /// directly predicts the noisy sample
        /// </summary>
        [EnumMember(Value = "sample")]
        Sample,

        /// <summary>
        /// see section 2.4 of Imagen Video paper
        /// </summary>
        [EnumMember(Value = "v-prediction")]
        V_Prediction
    }

    /// <summary>
    /// The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
    /// Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
    /// </summary>
    public enum Spacing {
        [EnumMember(Value = "leading")]
        Leading,
        [EnumMember(Value = "trailing")]
        Trailing,
        [EnumMember(Value = "linspace")]
        Linspace
    }

    /// <summary>
    /// The interpolation type to compute intermediate sigmas for the scheduler denoising steps.
    /// </summary>
    public enum Interpolation {
        [EnumMember(Value = "linear")]
        Linear,
        [EnumMember(Value = "log_linear")]
        LogLinear,
    }

    public enum Timestep {
        [EnumMember(Value = "discrete")]
        Discrete,
        [EnumMember(Value = "continuous")]
        Continuous,
    }

    public partial class SchedulerConfig : Config {

        public new static string ConfigName => "scheduler_config";


        [JsonProperty("beta_end")]
        public float? BetaEnd { get; set; }

        [JsonProperty("beta_schedule")]
        public Schedule? BetaSchedule { get; set; }

        [JsonProperty("beta_start")]
        public float? BetaStart { get; set; }

        [JsonProperty("clip_sample")]
        public bool? ClipSample { get; set; }

        [JsonProperty("clip_sample_range")]
        public float? ClipSampleRange { get; set; }

        [JsonProperty("dynamic_thresholding_ratio")]
        public float? DynamicThresholdingRatio { get; set; }

        [JsonProperty("interpolation_type")]
        public Interpolation? InterpolationType { get; set; }

        [JsonProperty("num_train_timesteps")]
        public int? NumTrainTimesteps { get; set; }

        [JsonProperty("prediction_type")]
        public Prediction? PredictionType { get; set; }
        
        [JsonProperty("rescale_betas_zero_snr")]
        public bool? RescaleBetasZeroSnr { get; set; }

        [JsonProperty("sample_max_value")]
        public float? SampleMaxValue { get; set; }

        [JsonProperty("set_alpha_to_one")]
        public bool? SetAlphaToOne { get; set; }

        [JsonProperty("skip_prk_steps")]
        public bool? SkipPrkSteps { get; set; }

        [JsonProperty("steps_offset")]
        public int? StepsOffset { get; set; }

        [JsonProperty("thresholding")]
        public bool? Thresholding { get; set; }

        [JsonProperty("timestep_spacing")]
        public Spacing? TimestepSpacing { get; set; }

        [JsonProperty("trained_betas")]
        public float[] TrainedBetas { get; set; }

        [JsonProperty("use_karras_sigmas")]
        public bool? UseKarrasSigmas { get; set; }

        [JsonProperty("sigma_min")]
        public float? SigmaMin { get; set; }

        [JsonProperty("sigma_max")]
        public float? SigmaMax { get; set; }

        [JsonProperty("timestep_type")]
        public Timestep? TimestepType { get; set; }
    }
}
