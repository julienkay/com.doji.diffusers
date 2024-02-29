using Newtonsoft.Json;

namespace Doji.AI.Diffusers {

    public class UnetConfig : Config {


        [JsonProperty("act_fn")]
        public string ActFn { get; set; }

        [JsonProperty("addition_embed_type")]
        public string AdditionEmbedType { get; set; }

        [JsonProperty("addition_embed_type_num_heads")]
        public int? AdditionEmbedTypeNumHeads { get; set; }

        [JsonProperty("addition_time_embed_dim")]
        public int? AdditionTimeEmbedDim { get; set; }

        [JsonProperty("attention_head_dim")]
        public int?[] AttentionHeadDim { get; set; }

        [JsonProperty("block_out_channels")]
        public int?[] BlockOutChannels { get; set; }

        [JsonProperty("center_input_sample")]
        public bool? CenterInputSample { get; set; }

        [JsonProperty("class_embed_type")]
        public object ClassEmbedType { get; set; }

        [JsonProperty("class_embeddings_concat")]
        public bool? ClassEmbeddingsConcat { get; set; }

        [JsonProperty("conv_in_kernel")]
        public int? ConvInKernel { get; set; }

        [JsonProperty("conv_out_kernel")]
        public int? ConvOutKernel { get; set; }

        [JsonProperty("cross_attention_dim")]
        public int? CrossAttentionDim { get; set; }

        [JsonProperty("cross_attention_norm")]
        public object CrossAttentionNorm { get; set; }

        [JsonProperty("down_block_types")]
        public string[] DownBlockTypes { get; set; }

        [JsonProperty("downsample_padding")]
        public int? DownsamplePadding { get; set; }

        [JsonProperty("dual_cross_attention")]
        public bool? DualCrossAttention { get; set; }

        [JsonProperty("encoder_hid_dim")]
        public object EncoderHidDim { get; set; }

        [JsonProperty("encoder_hid_dim_type")]
        public object EncoderHidDimType { get; set; }

        [JsonProperty("flip_sin_to_cos")]
        public bool? FlipSinToCos { get; set; }

        [JsonProperty("freq_shift")]
        public int? FreqShift { get; set; }

        [JsonProperty("in_channels")]
        public int? InChannels { get; set; }

        [JsonProperty("layers_per_block")]
        public int? LayersPerBlock { get; set; }

        [JsonProperty("mid_block_only_cross_attention")]
        public object MidBlockOnlyCrossAttention { get; set; }

        [JsonProperty("mid_block_scale_factor")]
        public int? MidBlockScaleFactor { get; set; }

        [JsonProperty("mid_block_type")]
        public string MidBlockType { get; set; }

        [JsonProperty("norm_eps")]
        public double NormEps { get; set; }

        [JsonProperty("norm_num_groups")]
        public int? NormNumGroups { get; set; }

        [JsonProperty("num_attention_heads")]
        public object NumAttentionHeads { get; set; }

        [JsonProperty("num_class_embeds")]
        public object NumClassEmbeds { get; set; }

        [JsonProperty("only_cross_attention")]
        public bool? OnlyCrossAttention { get; set; }

        [JsonProperty("out_channels")]
        public int? OutChannels { get; set; }

        [JsonProperty("projection_class_embeddings_input_dim")]
        public int? ProjectionClassEmbeddingsInputDim { get; set; }

        [JsonProperty("resnet_out_scale_factor")]
        public float? ResnetOutScaleFactor { get; set; }

        [JsonProperty("resnet_skip_time_act")]
        public bool? ResnetSkipTimeAct { get; set; }

        [JsonProperty("resnet_time_scale_shift")]
        public string ResnetTimeScaleShift { get; set; }

        [JsonProperty("sample_size")]
        public int SampleSize { get; set; }

        [JsonProperty("time_cond_proj_dim")]
        public object TimeCondProjDim { get; set; }

        [JsonProperty("time_embedding_act_fn")]
        public object TimeEmbeddingActFn { get; set; }

        [JsonProperty("time_embedding_dim")]
        public object TimeEmbeddingDim { get; set; }

        [JsonProperty("time_embedding_type")]
        public string TimeEmbeddingType { get; set; }

        [JsonProperty("timestep_post_act")]
        public object TimestepPostAct { get; set; }

        [JsonProperty("transformer_layers_per_block")]
        public int?[] TransformerLayersPerBlock { get; set; }

        [JsonProperty("up_block_types")]
        public string[] UpBlockTypes { get; set; }

        [JsonProperty("upcast_attention")]
        public object UpcastAttention { get; set; }

        [JsonProperty("use_linear_projection")]
        public bool? UseLinearProjection { get; set; }
    }
}