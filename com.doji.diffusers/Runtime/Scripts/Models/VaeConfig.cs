using Newtonsoft.Json;

namespace Doji.AI.Diffusers {

    public class VaeConfig : IConfig {

        [JsonProperty("act_fn")]
        public string ActFn { get; set; }

        [JsonProperty("block_out_channels")]
        public int[] BlockOutChannels { get; set; }

        [JsonProperty("down_block_types")]
        public string[] DownBlockTypes { get; set; }

        [JsonProperty("force_upcast")]
        public bool ForceUpcast { get; set; }

        [JsonProperty("in_channels")]
        public int InChannels { get; set; }

        [JsonProperty("latent_channels")]
        public int LatentChannels { get; set; }

        [JsonProperty("layers_per_block")]
        public int LayersPerBlock { get; set; }

        [JsonProperty("norm_num_groups")]
        public int NormNumGroups { get; set; }

        [JsonProperty("out_channels")]
        public int OutChannels { get; set; }

        [JsonProperty("sample_size")]
        public int SampleSize { get; set; }

        [JsonProperty("scaling_factor")]
        public float? ScalingFactor { get; set; }

        [JsonProperty("up_block_types")]
        public string[] UpBlockTypes { get; set; }
    }
}
