using Newtonsoft.Json;

namespace Doji.AI.Diffusers {

    public class TextEncoderConfig {

        [JsonProperty("architectures")]
        public string[] Architectures { get; set; }

        [JsonProperty("attention_dropout")]
        public float AttentionDropout { get; set; }

        [JsonProperty("bos_token_id")]
        public int BosTokenId { get; set; }

        [JsonProperty("dropout")]
        public float Dropout { get; set; }

        [JsonProperty("eos_token_id")]
        public int EosTokenId { get; set; }

        [JsonProperty("hidden_act")]
        public string HiddenAct { get; set; }

        [JsonProperty("hidden_size")]
        public int HiddenSize { get; set; }

        [JsonProperty("initializer_factor")]
        public float InitializerFactor { get; set; }

        [JsonProperty("initializer_range")]
        public float InitializerRange { get; set; }

        [JsonProperty("intermediate_size")]
        public int IntermediateSize { get; set; }

        [JsonProperty("layer_norm_eps")]
        public double LayerNormEps { get; set; }

        [JsonProperty("max_position_embeddings")]
        public int MaxPositionEmbeddings { get; set; }

        [JsonProperty("model_type")]
        public string ModelType { get; set; }

        [JsonProperty("num_attention_heads")]
        public int NumAttentionHeads { get; set; }

        [JsonProperty("num_hidden_layers")]
        public int NumHiddenLayers { get; set; }

        [JsonProperty("pad_token_id")]
        public int PadTokenId { get; set; }

        [JsonProperty("projection_dim")]
        public int ProjectionDim { get; set; }

        [JsonProperty("torch_dtype")]
        public string TorchDtype { get; set; }

        [JsonProperty("transformers_version")]
        public string TransformersVersion { get; set; }

        [JsonProperty("vocab_size")]
        public int VocabSize { get; set; }
    }
}