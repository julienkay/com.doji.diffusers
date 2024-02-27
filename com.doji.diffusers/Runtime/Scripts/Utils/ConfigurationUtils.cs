using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;
using System;
using System.IO;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Base class for all configuration classes.
    /// </summary>
    [JsonConverter(typeof(IConfigConverter))]
    public abstract class IConfig {

        [JsonProperty("_class_name")]
        public string ClassName { get; set; }

        [JsonProperty("_diffusers_version")]
        public string DiffusersVersion { get; set; }

        //public abstract static string Serialize(IConfig config);
        public static IConfig Deserialize(string config) {
            return Newtonsoft.Json.JsonConvert.DeserializeObject<IConfig>(config);
        }

        /*public static IConfig LoadConfig(string path) {
            return LoadConfigFromTextAsset(Path.Combine(path, ConfigName));
        }*/

    }

    public class IConfigSpecifiedConcreteClassConverter : DefaultContractResolver {
        protected override JsonConverter ResolveContractConverter(Type objectType) {
            if (typeof(IConfig).IsAssignableFrom(objectType) && !objectType.IsAbstract)
                return null; // pretend TableSortRuleConvert is not specified (thus avoiding a stack overflow)
            return base.ResolveContractConverter(objectType);
        }
    }


    public class IConfigConverter : JsonConverter {
        static JsonSerializerSettings SpecifiedSubclassConversion = new JsonSerializerSettings() { ContractResolver = new IConfigSpecifiedConcreteClassConverter() };

        public override bool CanConvert(Type objectType) {
            return objectType == typeof(IConfig);
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
            throw new NotImplementedException(); // won't be called because CanWrite returns false
        }
    }

    /// <summary>
    /// All configuration parameters are stored under `self.config`. Also
    /// provides the[`~ConfigMixin.from_config`] and[`~ConfigMixin.save_config`] methods for loading
    /// classes that inherit from <see cref="IConfigurable{T}"/>.
    /// </summary>
    public interface IConfigurable {
        public IConfig IConfig { get; }

        /*public static implicit operator IConfigurable<IConfig>(IConfigurable<T> obj) {
            return (IConfigurable<IConfig>)obj;
        }*/

        //public IConfigurable<T> FromPretrained();

        /*public static object FromPretrained(string path, BackendType backend) {
            var config = IConfig.LoadConfig(path);
            return FromConfig(config, backend);
        }

        /// <summary>
        /// Instantiate a class from an <see cref="IConfig"/> object.
        /// </summary>
        public static object FromConfig(IConfig config, BackendType backend) {
            Type type = Type.GetType("Doji.AI.Diffusers." + config.ClassName) ?? throw new Exception($"Unknown class type in config: {config.ClassName}");
            try {
                return Activator.CreateInstance(type, config, backend);
            } catch (Exception e) {
                Debug.LogError($"{e.GetType().Name} when trying to create class of type '{config.ClassName}'");
                throw e;
            }
        }*/

        /*public static IConfig LoadConfigFromTextAsset(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path)
                ?? throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            IConfig deserializedObject = JsonConvert.DeserializeObject<IConfig>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        public virtual IConfig LoadConfig(string path) {
            return LoadConfigFromTextAsset(Path.Combine(path, Path.GetFileNameWithoutExtension(ConfigName)));
        }

        public static IConfigurable FromPretrained(Type type, string path, BackendType backend) {

            IConfigurable x = (IConfigurable)Activator.CreateInstance(type);
            var config = x.LoadConfig(path);
            return x;
        }

        public static IConfigurable FromConfig(Type type) {
            //Type type = Type.GetType("Doji.AI.Diffusers." + config.ClassName) ?? throw new Exception($"Unknown class type in config: {type.Name}");
            try {
                return (IConfigurable)Activator.CreateInstance(type);
            } catch (Exception e) {
                Debug.LogError($"{e.GetType().Name} when trying to create class of type '{type.Name}'");
                throw e;
            }
        }*/

        public static IConfig LoadConfigFromTextAsset(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path)
                ?? throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            IConfig deserializedObject = JsonConvert.DeserializeObject<IConfig>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        public static IConfig LoadConfig(string path, string configName) {
            return LoadConfigFromTextAsset(Path.Combine(path, configName));
        }

        public static C FromConfig<C>(IConfig config, BackendType backend) where C : IConfigurable{
            Type type = Type.GetType("Doji.AI.Diffusers." + config.ClassName) ?? throw new Exception($"Unknown class type in config: {config.ClassName}");
            try {
                return (C)Activator.CreateInstance(type, config, backend);
            } catch (Exception e) {
                Debug.LogError($"{e.GetType().Name} when trying to create class of type '{type.Name}'");
                throw e;
            }
        }
    }
}