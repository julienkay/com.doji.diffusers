using Doji.AI.Transformers;
using Newtonsoft.Json;
using System;
using System.IO;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public partial class StableDiffusionPipeline {

        internal static PipelineConfig LoadPipelineConfig(DiffusionModel model) {
            return LoadJsonFromTextAsset<PipelineConfig>(Path.Combine(model.Name, "model_index"));
        }

        internal static VaeConfig LoadVaeConfig(string subFolder) {
            string path = Path.Combine(subFolder, "vae_decoder", "config");
            return LoadJsonFromTextAsset<VaeConfig>(path);
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
            
            PipelineConfig config = LoadPipelineConfig(model);

            /*string modelDir = model.Name;
            var subModelsToLoad = config.Components.Keys.Intersect(new string[] { "tokenizer", "tokenizer_2", "scheduler" });
            foreach (var name in subModelsToLoad) {
                string className = config.Components[name][1];
                className = ResolveClassName(name, className);
                Type type = Type.GetType("Doji.AI.Diffusers." + className) ?? throw new NotImplementedException($"Unknown pipeline component: {className}");
                if (!typeof(IConfigurable).IsAssignableFrom(type)) {
                    throw new ArgumentException($"Can not load component {name} from config.");
                }
                string path = Path.Combine(modelDir, name);
                var component = IConfigurable.FromPretrained(type, path, backend);
                Scheduler x = Scheduler.FromPretrained(path, backend);
            }

            return null;*/

            var vocab = LoadVocab(model.Name);
            var merges = LoadMerges(model.Name);
            var tokenizerConfig = LoadTokenizerConfig(model.Name);
            var clipTokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var schedulerConfig = LoadSchedulerConfig(model.Name);
            var scheduler = Scheduler.FromPretrained(Path.Combine(model.Name, "scheduler"), backend);
            var vaeConfig = LoadVaeConfig(model.Name);
            var vaeDecoder = new VaeDecoder(LoadVaeDecoder(model.Name), vaeConfig, backend);
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
            sdPipeline.Config = config;
            return sdPipeline;
        }

        /// <summary>
        /// Translates class names from a model_index.json into class names used in this Unity diffusers package.
        /// </summary>
        /// <remarks>
        /// Wish the whole "model" concept & ONNX conversions were more standardized.
        /// onnx export can be don through either huggingface/diffusers export scripts or huggingface/optimum. 
        /// Then some repos throw the onnx files in with the other model files without changing the class names in model_index.
        /// </remarks>
        private static string ResolveClassName(string subModel, string className) {
            switch (className) {
                case "OnnxRuntimeModel":
                    switch (subModel) {
                        case "text_encoder":
                            return "TextEncoder";
                        case "text_encoder_2":
                            return "TextEncoder";
                        case "unet":
                            return "Unet";
                        case "vae_decoder":
                            return "VaeDecoder";
                        case "vae_encoder":
                            return "VaeEncoder";
                        default:
                            return className;
                    }
                case "CLIPTextModel":
                    return "TextEncoder";
                case "CLIPTextModelWithProjection":
                    return "TextEncoder";
                case "UNet2DConditionModel":
                    return "Unet";
                case "AutoencoderKL":
                    return "VaeDecoder";
                default:
                    return className;
            }
        }

        /// <summary>
        /// Creates a Scheduler of the correct subclass based on the given <paramref name="config"/>.
        /// TODO: Might need to tag pipeline & scheduler classes or constructors with [UnityEngine.Scripting.Preserve]
        /// attribute for IL2CPP code stripping.
        /// </summary>
        /*private static DiffusionPipeline CreatePipeline(PipelineConfig config, BackendType backend) {
            Type type = Type.GetType(config.ClassName) ?? throw new NotImplementedException($"Unknown diffusion pipeline in config: {config.ClassName}");
            try {
                return (DiffusionPipeline)Activator.CreateInstance(type, config, backend);
            } catch (Exception e) {
                Debug.LogError($"{e.GetType().Name} when trying to create '{config.ClassName}'");
                throw e;
            }
        }

        /// <summary>
        /// Creates a Scheduler of the correct subclass based on the given <paramref name="config"/>.
        /// </summary>
        private static Scheduler CreateScheduler(SchedulerConfig config, BackendType backend) {
            Type type = Type.GetType(config.ClassName) ?? throw new NotImplementedException($"Unknown scheduler type in config: {config.ClassName}");
            try {
                return (Scheduler)Activator.CreateInstance(type, config, backend);
            } catch (Exception e) {
                Debug.LogError($"{e.GetType().Name} when trying to create scheduler of type '{config.ClassName}'");
                throw e;
            }
        }*/

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