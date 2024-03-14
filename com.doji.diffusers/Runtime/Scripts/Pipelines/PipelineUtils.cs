using Doji.AI.Transformers;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public partial class DiffusionPipeline {

        internal static PipelineConfig LoadPipelineConfig(DiffusionModel model) {
            return LoadJsonFromTextAsset<PipelineConfig>(Path.Combine(model.ModelId, "model_index"));
        }

        internal static VaeConfig LoadVaeConfig(string modelName) {
            string path = Path.Combine(modelName, "vae_decoder", "config");
            return LoadJsonFromTextAsset<VaeConfig>(path);
        }

        internal static Model LoadVaeDecoder(string modelName) {
            string path = Path.Combine(modelName, "vae_decoder", "model");
            return LoadFromModelAsset(path);
        }

        internal static Model LoadTextEncoder(string modelName) {
            string path = Path.Combine(modelName, "text_encoder", "model");
            return LoadFromModelAsset(path);
        }

        internal static Vocab LoadVocab(string modelName, string subFolder) {
            string path = Path.Combine(modelName, subFolder, "vocab");
            TextAsset vocabFile = Resources.Load<TextAsset>(path);
            var vocab = Vocab.Deserialize(vocabFile.text);
            Resources.UnloadAsset(vocabFile);
            return vocab;
        }

        internal static string LoadMerges(string modelName, string subFolder) {
            string path = Path.Combine(modelName, subFolder, "merges");
            return LoadFromTextAsset(path);
        }

        internal static TokenizerConfig LoadTokenizerConfig(string modelName, string subFolder) {
            string path = Path.Combine(modelName, subFolder, "tokenizer_config");
            return LoadJsonFromTextAsset<TokenizerConfig>(path);
        }

        internal static SchedulerConfig LoadSchedulerConfig(string modelName) {
            string path = Path.Combine(modelName, "scheduler", "scheduler_config");
            return LoadJsonFromTextAsset<SchedulerConfig>(path);
        }

        internal static Model LoadUnet(string modelName) {
            string path = Path.Combine(modelName, "unet", "model");
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

        public static DiffusionPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
#if UNITY_EDITOR
            OnModelRequested?.Invoke(model);
#endif

            PipelineConfig config = LoadPipelineConfig(model);
            return config.ClassName switch {
                "StableDiffusionPipeline"     => StableDiffusionPipeline.FromPretrained(model, backend),
                "OnnxStableDiffusionPipeline" => StableDiffusionPipeline.FromPretrained(model, backend),
                "StableDiffusionXLPipeline"   => StableDiffusionXLPipeline.FromPretrained(model, backend),
                _ => throw new NotImplementedException($"Unknown diffusion pipeline in config: {config.ClassName}"),
            };
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

        public static bool ExistsInResources(DiffusionModel model) {
            // check if at least all required files are present in Resources
            return model.All(file => !file.Required || File.Exists(file.ResourcesFilePath));
        }

        public static bool ExistsInStreamingAssets(DiffusionModel model) {
            foreach (var f in model) {
                bool r = f.Required;
                bool d = File.Exists(f.StreamingAssetsPath);

            }
            // check if at least all required files are present in StreamingAssets
            return model.All(file => !file.Required || File.Exists(file.StreamingAssetsPath));
        }
    }

    public partial class StableDiffusionPipeline {

        internal static new StableDiffusionPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
            PipelineConfig config = LoadPipelineConfig(model);

            var vocab = LoadVocab(model.ModelId, "tokenizer");
            var merges = LoadMerges(model.ModelId, "tokenizer");
            var tokenizerConfig = LoadTokenizerConfig(model.ModelId, "tokenizer");
            var clipTokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var scheduler = Scheduler.FromPretrained(model.ModelId, "scheduler", backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model.ModelId, "vae_decoder", backend);
            var textEncoder = TextEncoder.FromPretrained(model.ModelId, "text_encoder", backend);
            var unet = Unet.FromPretrained(model.ModelId, "unet", backend);

            StableDiffusionPipeline sdPipeline = new StableDiffusionPipeline(
                vaeDecoder,
                textEncoder,
                clipTokenizer,
                scheduler,
                unet,
                backend
            );
            sdPipeline.NameOrPath = model.ModelId;
            sdPipeline.Config = config;
            return sdPipeline;
        }
    }

    public partial class StableDiffusionXLPipeline {

        internal static new StableDiffusionXLPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
            PipelineConfig config = LoadPipelineConfig(model);

            var vocab = LoadVocab(model.ModelId, "tokenizer");
            var merges = LoadMerges(model.ModelId, "tokenizer");
            var tokenizerConfig = LoadTokenizerConfig(model.ModelId, "tokenizer");
            var tokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            vocab = LoadVocab(model.ModelId, "tokenizer_2");
            merges = LoadMerges(model.ModelId, "tokenizer_2");
            tokenizerConfig = LoadTokenizerConfig(model.ModelId, "tokenizer_2");
            var tokenizer2 = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var scheduler = Scheduler.FromPretrained(model.ModelId, "scheduler", backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model.ModelId, "vae_decoder", backend);
            var textEncoder = TextEncoder.FromPretrained(model.ModelId, "text_encoder", backend);
            var textEncoder2 = TextEncoder.FromPretrained(model.ModelId, "text_encoder_2", backend);
            var unet = Unet.FromPretrained(model.ModelId, "unet", backend);

            StableDiffusionXLPipeline sdPipeline = new StableDiffusionXLPipeline(
                vaeDecoder,
                textEncoder,
                tokenizer,
                scheduler,
                unet,
                textEncoder2,
                tokenizer2,
                backend
            );
            sdPipeline.NameOrPath = model.ModelId;
            sdPipeline.Config = config;
            return sdPipeline;
        }
    }
}