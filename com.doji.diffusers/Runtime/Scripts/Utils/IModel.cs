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
                return null;
            }
            Model model = ModelLoader.Load(modelAsset);
            Resources.UnloadAsset(modelAsset);
            return model;
        }

        /// <summary>
        /// Load a Model from either StreamingAssets or Resources.
        /// If no config is found null is returned
        /// </summary>
        protected static Model LoadModel(ModelFile file) {
            if (File.Exists(file.StreamingAssetsPath)) {
                return ModelLoader.Load(file.StreamingAssetsPath);
            }
            return LoadFromModelAsset(file.ResourcePath);
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

        protected static C FromPretrained<C>(ModelFile modelFile, ModelFile modelConfig, BackendType backend) where C : IModel<T> {
            var config = LoadConfig(modelConfig) ?? throw new FileNotFoundException($"File '{modelConfig.FileName}' not found for: '{typeof(T).Name}'");
            var model = LoadModel(modelFile) ?? throw new FileNotFoundException($"File '{modelFile.FileName}' not found for: '{typeof(T).Name}'");
            return FromConfig<C>(config, model, backend);
        }
    }
}