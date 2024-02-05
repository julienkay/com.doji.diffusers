using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace Doji.AI.Diffusers {

    public enum Schedule {
        [EnumMember(Value = "linear")]
        Linear,

        [EnumMember(Value = "scaled_linear")]
        ScaledLinear,

        [EnumMember(Value = "squaredcos_cap_v2")]
        SquaredCosCapV2
    }

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

    public enum Spacing {
        [EnumMember(Value = "leading")]
        Leading,
        [EnumMember(Value = "trailing")]
        Trailing,
        [EnumMember(Value = "linspace")]
        Linspace
    }

    public partial class SchedulerConfig {
        [JsonProperty("_class_name")]
        public string ClassName { get; set; }

        [JsonProperty("_diffusers_version")]
        public string DiffusersVersion { get; set; }

        [JsonProperty("beta_end")]
        public float BetaEnd { get; set; }

        [JsonProperty("beta_schedule")]
        public Schedule BetaSchedule { get; set; }

        [JsonProperty("beta_start")]
        public float BetaStart { get; set; }

        [JsonProperty("clip_sample")]
        public bool ClipSample { get; set; }

        [JsonProperty("clip_sample_range")]
        public float ClipSampleRange { get; set; }

        [JsonProperty("dynamic_thresholding_ratio")]
        public float DynamicThresholdingRatio { get; set; }

        [JsonProperty("num_train_timesteps")]
        public int NumTrainTimesteps { get; set; }

        [JsonProperty("prediction_type")]
        public Prediction PredictionType { get; set; }
        
        [JsonProperty("rescale_betas_zero_snr")]
        public bool RescaleBetasZeroSnr { get; set; }

        [JsonProperty("sample_max_value")]
        public float SampleMaxValue { get; set; }

        [JsonProperty("set_alpha_to_one")]
        public bool SetAlphaToOne { get; set; }

        [JsonProperty("skip_prk_steps")]
        public bool SkipPrkSteps { get; set; }

        [JsonProperty("steps_offset")]
        public int StepsOffset { get; set; }

        [JsonProperty("thresholding")]
        public bool Thresholding { get; set; }

        [JsonProperty("timestep_spacing")]
        public Spacing TimestepSpacing { get; set; }

        [JsonProperty("trained_betas")]
        public float[] TrainedBetas { get; set; }
    }
}
