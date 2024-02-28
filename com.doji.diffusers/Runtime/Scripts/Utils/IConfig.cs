using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;
using Newtonsoft.Json;
using System;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Base class for all configuration classes.
    /// </summary>
    [JsonConverter(typeof(ConfigConverter))]
    public abstract class Config {

        /// <summary>
        /// The file name of the config file (without extension, .json implicitly assumed)
        /// </summary>
        public static string ConfigName => "config";

        [JsonProperty("_class_name")]
        public string ClassName { get; set; }

        [JsonProperty("_diffusers_version")]
        public string DiffusersVersion { get; set; }
    }

    public class ConfigSpecifiedConcreteClassConverter : DefaultContractResolver {
        protected override JsonConverter ResolveContractConverter(Type objectType) {
            if (typeof(Config).IsAssignableFrom(objectType) && !objectType.IsAbstract)
                return null; // (avoiding a stack overflow)
            return base.ResolveContractConverter(objectType);
        }
    }

    public class ConfigConverter : JsonConverter {
        static JsonSerializerSettings SpecifiedSubclassConversion = new JsonSerializerSettings() { ContractResolver = new ConfigSpecifiedConcreteClassConverter() };

        public override bool CanConvert(Type objectType) {
            return objectType == typeof(Config);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
            JObject jo = JObject.Load(reader);
            string configType = jo["_class_name"].Value<string>();
            if (configType.Contains("Scheduler")) {
                return JsonConvert.DeserializeObject<SchedulerConfig>(jo.ToString(), SpecifiedSubclassConversion);
            } else {
                throw new Exception($"Unknown class type in config: {configType}");
            }
        }

        public override bool CanWrite {
            get { return false; }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) {
            throw new NotImplementedException();
        }
    }
}