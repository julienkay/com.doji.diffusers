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

        [JsonProperty("num_train_timesteps")]
        public int NumTrainTimesteps { get; set; }

        [JsonProperty("set_alpha_to_one")]
        public bool SetAlphaToOne { get; set; }

        [JsonProperty("skip_prk_steps")]
        public bool SkipPrkSteps { get; set; }

        [JsonProperty("steps_offset")]
        public int StepsOffset { get; set; }

        [JsonProperty("trained_betas")]
        public float[] TrainedBetas { get; set; }
    }
}
