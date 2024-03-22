using Newtonsoft.Json;
using System;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Metadata containing information about which pipeline/model/settings were used
    /// to generate an image.
    /// </summary>
    public class Metadata {
        [JsonProperty("comment")]
#pragma warning disable IDE0051
        private const string COMMENT = "This image was generated using https://github.com/julienkay/com.doji.diffusers";
#pragma warning restore IDE0051

        [JsonProperty("package_version")]
        [JsonConverter(typeof(PackageVersionConverter))]
        public string PackageVersion { get; set; }

        [JsonProperty("model")]
        public string Model { get; set; }

        [JsonProperty("pipeline")]
        public string Pipeline { get; set; }

        [JsonProperty("sampler")]
        public string Sampler { get; set; }

        [JsonProperty("parameters")]
        public Parameters Parameters { get; set; }

        public string Serialize() {
            return JsonConvert.SerializeObject(this);
        }

        public static Metadata Deserialize(string data) {
            return JsonConvert.DeserializeObject<Metadata>(data);
        }
    }

    public class PackageVersionConverter : JsonConverter {
        public override bool CanConvert(Type objectType) {
            return objectType == typeof(string);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
            return serializer.Deserialize<string>(reader);
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) {
            string packageVersion = System.Diagnostics.FileVersionInfo.GetVersionInfo(System.Reflection.Assembly.GetExecutingAssembly().Location).ProductVersion;
            writer.WriteValue(packageVersion.ToString());
        }
    }
}