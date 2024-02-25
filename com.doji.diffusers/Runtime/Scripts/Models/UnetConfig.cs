using Newtonsoft.Json;

namespace Doji.AI.Diffusers {

    public class UnetConfig {

        [JsonProperty("_class_name")]
        public string ClassName { get; set; }

        [JsonProperty("_diffusers_version")]
        public string DiffusersVersion { get; set; }

        [JsonProperty("act_fn")]
        public string ActFn { get; set; }

        [JsonProperty("attention_head_dim")]
        public int AttentionHeadDim { get; set; }

        [JsonProperty("block_out_channels")]
        public int[] BlockOutChannels { get; set; }

        [JsonProperty("center_input_sample")]
        public bool CenterInputSample { get; set; }

        [JsonProperty("cross_attention_dim")]
        public int CrossAttentionDim { get; set; }

        [JsonProperty("down_block_types")]
        public string[] DownBlockTypes { get; set; }

        [JsonProperty("downsample_padding")]
        public int DownsamplePadding { get; set; }

        [JsonProperty("flip_sin_to_cos")]
        public bool FlipSinToCos { get; set; }

        [JsonProperty("freq_shift")]
        public int FreqShift { get; set; }

        [JsonProperty("in_channels")]
        public int InChannels { get; set; }

        [JsonProperty("layers_per_block")]
        public int LayersPerBlock { get; set; }

        [JsonProperty("mid_block_scale_factor")]
        public int MidBlockScaleFactor { get; set; }

        [JsonProperty("norm_eps")]
        public double NormEps { get; set; }

        [JsonProperty("norm_num_groups")]
        public int NormNumGroups { get; set; }

        [JsonProperty("out_channels")]
        public int OutChannels { get; set; }

        [JsonProperty("sample_size")]
        public int SampleSize { get; set; }

        [JsonProperty("up_block_types")]
        public string[] UpBlockTypes { get; set; }
    }
}