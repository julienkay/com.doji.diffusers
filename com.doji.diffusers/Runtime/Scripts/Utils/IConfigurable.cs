using Newtonsoft.Json;
using System.IO;
using System;
using UnityEngine;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// All configuration parameters are stored under <see cref="IConfigurable{T}.Config"/>.
    /// Also provides a <see cref="IConfigurable.FromConfig{C}(Config, BackendType)"/>
    /// method for loading classes that inherit from <see cref="IConfigurable{T}"/>.
    /// </summary>
    public interface IConfigurable<T> where T : Config {

        public T Config { get; }

        /// <summary>
        /// Load a config file from a Resources folder.
        /// </summary>
        private static Config LoadConfigFromTextAsset(string resourcePath) {
            TextAsset textAsset = Resources.Load<TextAsset>(resourcePath);
            if (textAsset == null) {
                //Debug.LogError($"The TextAsset file was not found at: '{path}'");
                return null;
            }

            Config deserializedObject = JsonConvert.DeserializeObject<Config>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        /// <summary>
        /// Load a config file from either StreamingAssets or Resources.
        /// If no config is found null is returned
        /// </summary>
        protected static Config LoadConfig(ModelFile file) {
            if (File.Exists(file.StreamingAssetsPath)) {
                return JsonConvert.DeserializeObject<Config>(File.ReadAllText(file.StreamingAssetsPath));
            }
            return LoadConfigFromTextAsset(file.ResourcePath);
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

        internal static C FromPretrained<C>(ModelFile file, BackendType backend) where C : IConfigurable<T> {
            var config = LoadConfig(file) ?? throw new FileNotFoundException($"File '{file.FileName}' not found for: '{typeof(T).Name}'");
            return FromConfig<C>(config, backend);
        }
    }
}