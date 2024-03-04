using System.IO;
using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public interface IModel<T> : IConfigurable<T> where T : Config {

        /// <summary>
        /// Loads a Sentis <see cref="Model"/> from a <see cref="ModelAsset"/> in Resources.
        /// </summary>
        /// <param name="path">The path to the model file in the Resources folder</param>
        private static Model LoadFromModelAsset(string path) {
            ModelAsset modelAsset = Resources.Load<ModelAsset>(path);
            if (modelAsset == null) {
                throw new FileNotFoundException($"The ModelAsset file was not found at: '{path}'");
            }
            Model model = ModelLoader.Load(modelAsset);
            Resources.UnloadAsset(modelAsset);
            return model;
        }

        private static C FromConfig<C>(Config config, Model model, BackendType backend) where C : IModel<T> {
            Type type = typeof(C) ?? throw new Exception($"Unknown class type in config: {config.ClassName}");
            try {
                return (C)Activator.CreateInstance(type, model, config, backend);
            } catch (Exception e) {
                Debug.LogError($"{e.GetType().Name} when trying to create class of type '{type.Name}'");
                throw e;
            }
        }

        protected new static C FromPretrained<C>(string modelDir, string subFolder, string configName, BackendType backend) where C : IModel<T> {
            var config = LoadConfig(Path.Combine(modelDir, subFolder), configName);
            var model = LoadFromModelAsset(Path.Combine(modelDir, subFolder, "model"));
            return FromConfig<C>(config, model, backend);
        }
    }
}