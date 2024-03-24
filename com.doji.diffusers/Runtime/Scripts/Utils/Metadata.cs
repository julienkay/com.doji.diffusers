using Newtonsoft.Json;
using System;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Metadata containing information about which pipeline/model/settings were used
    /// to generate an image.
    /// </summary>
    public class Metadata {

        [JsonProperty("comment")]
        public readonly string Comment = "This image was generated using https://github.com/julienkay/com.doji.diffusers";

        [JsonProperty("package_version")]
        public readonly string PackageVersion = System.Diagnostics.FileVersionInfo.GetVersionInfo(System.Reflection.Assembly.GetExecutingAssembly().Location).ProductVersion;

        [JsonProperty("model")]
        public string Model { get; set; }

        [JsonProperty("pipeline")]
        public string Pipeline { get; set; }

        [JsonProperty("sampler")]
        public string Sampler { get; set; }

        [JsonProperty("parameters")]
        public Parameters Parameters { get; set; }

        public string Serialize() {
            return JsonConvert.SerializeObject(this, new JsonSerializerSettings {
                NullValueHandling = NullValueHandling.Ignore
            });
        }

        public static Metadata Deserialize(string data) {
            return JsonConvert.DeserializeObject<Metadata>(data);
        }
    }
}