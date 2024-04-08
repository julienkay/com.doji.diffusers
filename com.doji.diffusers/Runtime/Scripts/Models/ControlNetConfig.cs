using Newtonsoft.Json;

namespace Doji.AI.Diffusers {
    public class ControlNetConfig : Config {

        [JsonProperty("act_fn")]
        public string ActFn { get; set; }

        [JsonProperty("attention_head_dim")]
        public long AttentionHeadDim { get; set; }

        [JsonProperty("block_out_channels")]
        public long[] BlockOutChannels { get; set; }

        [JsonProperty("center_input_sample")]
        public bool CenterInputSample { get; set; }

        [JsonProperty("class_embed_type")]
        public object ClassEmbedType { get; set; }

        [JsonProperty("controlnet_conditioning_channels")]
        public long ControlnetConditioningChannels { get; set; }

        [JsonProperty("controlnet_conditioning_embedding_type")]
        public string ControlnetConditioningEmbeddingType { get; set; }

        [JsonProperty("conv_in_kernel")]
        public long ConvInKernel { get; set; }

        [JsonProperty("conditioning_embedding_out_channels")]
        public long[] ConditioningEmbeddingOutChannels { get; set; }

        [JsonProperty("controlnet_conditioning_channel_order")]
        public string ControlnetConditioningChannelOrder { get; set; }

        [JsonProperty("cross_attention_dim")]
        public long CrossAttentionDim { get; set; }

        [JsonProperty("down_block_types")]
        public string[] DownBlockTypes { get; set; }

        [JsonProperty("downsample_padding")]
        public long DownsamplePadding { get; set; }

        [JsonProperty("dual_cross_attention")]
        public bool DualCrossAttention { get; set; }

        [JsonProperty("flip_sin_to_cos")]
        public bool FlipSinToCos { get; set; }

        [JsonProperty("freq_shift")]
        public long FreqShift { get; set; }

        [JsonProperty("in_channels")]
        public long InChannels { get; set; }

        [JsonProperty("layers_per_block")]
        public long LayersPerBlock { get; set; }

        [JsonProperty("mid_block_scale_factor")]
        public long MidBlockScaleFactor { get; set; }

        [JsonProperty("mid_block_type")]
        public string MidBlockType { get; set; }

        [JsonProperty("norm_eps")]
        public double NormEps { get; set; }

        [JsonProperty("norm_num_groups")]
        public long NormNumGroups { get; set; }

        [JsonProperty("num_class_embeds")]
        public object NumClassEmbeds { get; set; }

        [JsonProperty("only_cross_attention")]
        public bool OnlyCrossAttention { get; set; }

        [JsonProperty("projection_class_embeddings_input_dim")]
        public object ProjectionClassEmbeddingsInputDim { get; set; }

        [JsonProperty("resnet_time_scale_shift")]
        public string ResnetTimeScaleShift { get; set; }

        [JsonProperty("sample_size")]
        public long SampleSize { get; set; }

        [JsonProperty("time_cond_proj_dim")]
        public object TimeCondProjDim { get; set; }

        [JsonProperty("time_embedding_type")]
        public string TimeEmbeddingType { get; set; }

        [JsonProperty("timestep_post_act")]
        public object TimestepPostAct { get; set; }

        [JsonProperty("upcast_attention")]
        public bool UpcastAttention { get; set; }

        [JsonProperty("use_linear_projection")]
        public bool UseLinearProjection { get; set; }

        [JsonProperty("global_pool_conditions")]
        public bool GlobalPoolConditions { get; set; }
    }
}