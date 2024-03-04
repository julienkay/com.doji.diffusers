using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;

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
            JToken className = jo["_class_name"];
            string configType;
            if (className == null) {
                // hacky... TextEncoder should be in transformers package and is a different kind of config
                configType = jo["architectures"].Value<JArray>().First.Value<string>();
            } else {
                configType = className.Value<string>();
            }
            if (configType.Contains("Scheduler")) {
                return JsonConvert.DeserializeObject<SchedulerConfig>(jo.ToString(), SpecifiedSubclassConversion);
            } else if (configType == "CLIPTextModel") {
                return JsonConvert.DeserializeObject<TextEncoderConfig>(jo.ToString(), SpecifiedSubclassConversion);
            } else if (configType == "CLIPTextModelWithProjection") {
                return JsonConvert.DeserializeObject<TextEncoderConfig>(jo.ToString(), SpecifiedSubclassConversion);
            } else if (configType == "UNet2DConditionModel") {
                return JsonConvert.DeserializeObject<UnetConfig>(jo.ToString(), SpecifiedSubclassConversion);
            } else if (configType == "AutoencoderKL") {
                return JsonConvert.DeserializeObject<VaeConfig>(jo.ToString(), SpecifiedSubclassConversion);
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