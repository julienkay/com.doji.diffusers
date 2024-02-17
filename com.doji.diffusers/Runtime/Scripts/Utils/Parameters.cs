using Newtonsoft.Json;
using System;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// All parameters and settings that were used to generate a given image.
    /// </summary>
    public class Parameters {
        [JsonProperty("comment")]
        private const string COMMENT = "This image was generated using https://github.com/julienkay/com.doji.diffusers";

        [JsonProperty("package_version")]
        public Version PackageVersion { get; set; }

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

        //[JsonProperty("seed")]
        //public int? Seed { get; set; }

        [JsonProperty("width")]
        public int Width { get; set; }

        [JsonProperty("height")]
        public int Height { get; set; }

        [JsonProperty("model")]
        public string Model { get; set; }

        [JsonProperty("eta")]
        public float? Eta { get; set; }

        public string Serialize() {
            return JsonConvert.SerializeObject(this);
        }
    }
}