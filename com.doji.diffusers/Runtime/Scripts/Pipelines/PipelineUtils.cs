using Doji.AI.Transformers;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public partial class StableDiffusionPipeline {

        internal static ModelIndex LoadModelIndex(DiffusionModel model) {
            return LoadJsonFromTextAsset<ModelIndex>(Path.Combine(model.Name, "model_index"));
        }

        internal static Model LoadVaeDecoder(string subFolder) {
            string path = Path.Combine(subFolder, "vae_decoder", "model");
            return LoadFromModelAsset(path);
        }

        internal static Model LoadTextEncoder(string subFolder) {
            string path = Path.Combine(subFolder, "text_encoder", "model");
            return LoadFromModelAsset(path);
        }

        internal static Vocab LoadVocab(string subFolder) {
            string path = Path.Combine(subFolder, "tokenizer", "vocab");
            TextAsset vocabFile = Resources.Load<TextAsset>(path);
            var vocab = Vocab.Deserialize(vocabFile.text);
            Resources.UnloadAsset(vocabFile);
            return vocab;
        }

        internal static string LoadMerges(string subFolder) {
            string path = Path.Combine(subFolder, "tokenizer", "merges");
            return LoadFromTextAsset(path);
        }

        internal static TokenizerConfig LoadTokenizerConfig(string subFolder) {
            string path = Path.Combine(subFolder, "tokenizer", "tokenizer_config");
            return LoadJsonFromTextAsset<TokenizerConfig>(path);
        }

        internal static SchedulerConfig LoadSchedulerConfig(string subFolder) {
            string path = Path.Combine(subFolder, "scheduler", "scheduler_config");
            return LoadJsonFromTextAsset<SchedulerConfig>(path);
        }

        internal static Model LoadUnet(string subFolder) {
            string path = Path.Combine(subFolder, "unet", "model");
            return LoadFromModelAsset(path);
        }

        /// <summary>
        /// Loads a string from a <see cref="TextAsset"/> in Resources.
        /// </summary>
        /// <param name="path">The path to the text file in the Resources folder</param>
        private static string LoadFromTextAsset(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path)
                ?? throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            string text = textAsset.text;
            Resources.UnloadAsset(textAsset);
            return text;
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a text asset in Resources
        /// by deserializing using <see cref="Newtonsoft.Json.JsonConvert"/>.
        /// </summary>
        /// <param name="path">The path to the text file in the Resources folder</param>
        private static T LoadJsonFromTextAsset<T>(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path)
                ?? throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            T deserializedObject = JsonConvert.DeserializeObject<T>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        /// <summary>
        /// Loads a Sentis <see cref="Model"/> from a <see cref="ModelAsset"/> in Resources.
        /// </summary>
        /// <param name="path">The path to the model file in the Resources folder</param>
        private static Model LoadFromModelAsset(string path) {
            ModelAsset modelAsset = Resources.Load<ModelAsset>(path)
                ?? throw new FileNotFoundException($"The ModelAsset file was not found at: '{path}'");
            Model model = ModelLoader.Load(modelAsset);
            Resources.UnloadAsset(modelAsset);
            return model;
        }

#if UNITY_EDITOR
        public static event Action<DiffusionModel> OnModelRequested = (x) => { };
#endif

        public static StableDiffusionPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
#if UNITY_EDITOR
            OnModelRequested?.Invoke(model);
#endif

            ModelIndex index = LoadModelIndex(model);
            var vocab = LoadVocab(model.Name);
            var merges = LoadMerges(model.Name);
            var tokenizerConfig = LoadTokenizerConfig(model.Name);
            var clipTokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var schedulerConfig = LoadSchedulerConfig(model.Name);
            var scheduler = CreateScheduler(schedulerConfig, backend: backend);
            var vaeDecoder = LoadVaeDecoder(model.Name);
            var textEncoder = LoadTextEncoder(model.Name);
            var unet = LoadUnet(model.Name);
            StableDiffusionPipeline sdPipeline = new StableDiffusionPipeline(
                vaeDecoder,
                textEncoder,
                clipTokenizer,
                scheduler,
                unet,
                backend
            );
            sdPipeline.NameOrPath = model.Name;
            sdPipeline.Config = index;
            return sdPipeline;
        }

        private static readonly Dictionary<string, Type> _schedulerTypes = new Dictionary<string, Type>() {
            { "DDIMScheduler", typeof(DDIMScheduler) },
            { "PNDMScheduler", typeof(PNDMScheduler) },
        };

        /// <summary>
        /// Creates a Scheduler of the correct subclass based on the given <paramref name="config"/>.
        /// TODO: Might need to tag scheduler classes or constructors with [UnityEngine.Scripting.Preserve]
        /// attribute for IL2CPP code stripping.
        /// </summary>
        private static Scheduler CreateScheduler(SchedulerConfig config, BackendType backend) {
            if (!_schedulerTypes.TryGetValue(config.ClassName, out Type type)) {
                throw new InvalidDataException($"Invalid/Unsupported scheduler type in config: {config.ClassName}");
            } else {
                try {
                    return (Scheduler)Activator.CreateInstance(type, config, backend);
                } catch (Exception e) {
                    Debug.LogError($"{e.GetType().Name} when trying to create scheduler of type '{config.ClassName}'");
                    throw e;
                }
            }
        }

        public static bool IsModelAvailable(DiffusionModel model) {
            if (ExistsInStreamingAssets(model)) {
                return true;
            }
            if (ExistsInResources(model)) {
                return true;
            }
            return false;
        }

        private static bool ExistsInResources(DiffusionModel model) {
            string dir = Path.GetDirectoryName(model.Name);
            string subDir = Path.GetFileName(model.Name);
            string path = Path.Combine(dir, subDir, "model_index");
            var modelIndex = Resources.Load<TextAsset>(path);
            bool exists = modelIndex != null;
            Resources.UnloadAsset(modelIndex);
            return exists;
        }

        private static bool ExistsInStreamingAssets(DiffusionModel model) {
            string path = Path.Combine(Application.streamingAssetsPath, Path.GetDirectoryName(model.Name));

            if (Directory.Exists(path)) {
                return true;
            } else {
                return false;
            }
        }
    }
}