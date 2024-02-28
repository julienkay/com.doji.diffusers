using Newtonsoft.Json;
using System.IO;
using System;
using UnityEngine;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// All configuration parameters are stored under <see cref="IConfigurable.IConfig"/>.
    /// Also provides a <see cref="IConfigurable.FromConfig{C}(Config, BackendType)"/>
    /// method for loading classes that inherit from <see cref="IConfigurable"/>.
    /// </summary>
    public interface IConfigurable<T> where T : Config {

        public T Config { get; }

        private static Config LoadConfigFromTextAsset(string modelDir) {
            TextAsset textAsset = Resources.Load<TextAsset>(modelDir);
            if (textAsset == null) {
                //Debug.LogError($"The TextAsset file was not found at: '{path}'");
                return null;
            }

            Config deserializedObject = JsonConvert.DeserializeObject<Config>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        protected static Config LoadConfig(string modelDir, string configName) {
            return LoadConfigFromTextAsset(Path.Combine(modelDir, configName));
        }

        private static C FromConfig<C>(Config config, BackendType backend) where C : IConfigurable<T> {
            Type type = Type.GetType("Doji.AI.Diffusers." + config.ClassName) ?? throw new Exception($"Unknown class type in config: {config.ClassName}");
            try {
                return (C)Activator.CreateInstance(type, config, backend);
            } catch (Exception e) {
                Debug.LogError($"{e.GetType().Name} when trying to create class of type '{type.Name}'");
                throw e;
            }
        }

        protected static C FromPretrained<C>(string modelDir, string subFolder, string configName, BackendType backend) where C : IConfigurable<T> {
            var config = LoadConfig(Path.Combine(modelDir, subFolder), configName);
            return config == null
                ? throw new FileNotFoundException($"No {configName}.json file was not found for: '{modelDir}'")
                : FromConfig<C>(config, backend);
        }
    }
}