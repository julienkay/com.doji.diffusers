using Newtonsoft.Json;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// All parameters and settings that were used to generate a given image.
    /// </summary>
    public class Parameters {
        [JsonProperty("comment")]
#pragma warning disable IDE0051
        private const string COMMENT = "This image was generated using https://github.com/julienkay/com.doji.diffusers";
#pragma warning restore IDE0051

        [JsonProperty("package_version")]
        public string PackageVersion { get; set; }

        [JsonProperty("model")]
        public string Model { get; set; }

        [JsonProperty("pipeline")]
        public string Pipeline { get; set; }

        [JsonProperty("prompt")]
        public string Prompt { get; set; }

        [JsonProperty("negative_prompt")]
        public string NegativePrompt { get; set; }

        [JsonProperty("steps")]
        public int Steps { get; set; }

        [JsonProperty("sampler")]
        public string Sampler { get; set; }

        [JsonProperty("cfg_scale")]
        public float CfgScale { get; set; }

        [JsonProperty("seed")]
        public uint? Seed { get; set; }

        [JsonProperty("width")]
        public int Width { get; set; }

        [JsonProperty("height")]
        public int Height { get; set; }

        [JsonProperty("eta")]
        public float? Eta { get; set; }

        public string Serialize() {
            return JsonConvert.SerializeObject(this, new JsonSerializerSettings {
                NullValueHandling = NullValueHandling.Ignore
            });
        }
    }
}