using Doji.AI.Transformers;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public partial class DiffusionPipeline {

        /** first try to load from StreamingAssets fall back to Resources **/

        internal static PipelineConfig LoadPipelineConfig(DiffusionModel model) {
            if (File.Exists(model.ModelIndex.StreamingAssetsPath)) {
                return LoadJsonFromFile<PipelineConfig>(model.ModelIndex.StreamingAssetsPath);
            }
            return LoadJsonFromTextAsset<PipelineConfig>(model.ModelIndex.ResourcePath);
        }

        internal static VaeConfig LoadVaeConfig(DiffusionModel model) {
            if (File.Exists(model.VaeDecoderConfig.StreamingAssetsPath)) {
                return LoadJsonFromFile<VaeConfig>(model.VaeDecoderConfig.StreamingAssetsPath);
            }
            return LoadJsonFromTextAsset<VaeConfig>(model.VaeDecoderConfig.ResourcePath);
        }

        internal static Model LoadVaeDecoder(DiffusionModel model) {
            if (File.Exists(model.VaeDecoder.StreamingAssetsPath)) {
                return LoadModel(model.VaeDecoder.StreamingAssetsPath);
            }
            return LoadFromModelAsset(model.VaeDecoder.ResourcePath);
        }

        internal static Model LoadTextEncoder(DiffusionModel model) {
            if (File.Exists(model.TextEncoder.StreamingAssetsPath)) {
                return LoadModel(model.TextEncoder.StreamingAssetsPath);
            }
            return LoadFromModelAsset(model.TextEncoder.ResourcePath);
        }

        internal static Vocab LoadVocab(DiffusionModel model) {
            return LoadFromJson<Vocab>(model.Vocab);
        }

        internal static string LoadMerges(DiffusionModel model) {
            return LoadText(model.Merges);
        }

        internal static TokenizerConfig LoadTokenizerConfig(DiffusionModel model) {
            return LoadFromJson<TokenizerConfig>(model.TokenizerConfig);
        }

        internal static Model LoadUnet(string modelName) {
            string path = Path.Combine(modelName, "unet", "model");
            return LoadFromModelAsset(path);
        }

        /// <summary>
        /// Loads a string from a text file either in StreamingAssets or Resources.
        /// </summary>
        internal static string LoadText(ModelFile file) {
            if (File.Exists(file.StreamingAssetsPath)) {
                return LoadTextFile(file.StreamingAssetsPath);
            }
            return LoadFromTextAsset(file.ResourcePath);
        }

        /// <summary>
        /// Loads a string from a <see cref="TextAsset"/> in Resources.
        /// </summary>
        /// <param name="path">The path to the text file in the Resources folder</param>
        private static string LoadFromTextAsset(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path);
            if (textAsset == null) {
                throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            }
            string text = textAsset.text;
            Resources.UnloadAsset(textAsset);
            return text;
        }

        /// <summary>
        /// Loads a string from from a json file in StreamingAssets
        /// </summary>
        /// <param name="path">The path to the text file in the Resources folder</param>
        private static string LoadTextFile(string path) {
#if !UNITY_STANDALONE
            throw new NotImplementedException();
#endif
            if (!File.Exists(path)) {
                throw new FileNotFoundException($"The text file was not found at: '{path}'");
            }
            return File.ReadAllText(path);
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a json file
        /// either in StreamingAssets or Resources.
        /// </summary>
        internal static T LoadFromJson<T>(ModelFile file) {
            if (File.Exists(file.StreamingAssetsPath)) {
                return LoadJsonFromFile<T>(file.StreamingAssetsPath);
            }
            return LoadJsonFromTextAsset<T>(file.ResourcePath);
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a text asset in Resources
        /// by deserializing using <see cref="Newtonsoft.Json.JsonConvert"/>.
        /// </summary>
        /// <param name="path">The path to the text file in the Resources folder</param>
        private static T LoadJsonFromTextAsset<T>(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path);
            if (textAsset == null) {
                throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            }
            T deserializedObject = JsonConvert.DeserializeObject<T>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        /// <summary>
        /// Loads an object of type <typeparamref name="T"/> from a json file
        /// by deserializing using <see cref="Newtonsoft.Json.JsonConvert"/>.
        /// </summary>
        private static T LoadJsonFromFile<T>(string path) {
#if !UNITY_STANDALONE
            throw new NotImplementedException();
#endif
            if (!File.Exists(path)) {
                throw new FileNotFoundException($"The .json file was not found at: '{path}'");
            }
            string json = File.ReadAllText(path);
            T deserializedObject = JsonConvert.DeserializeObject<T>(json);
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

        /// <summary>
        /// Loads a Sentis <see cref="Model"/> from StreamingAssets.
        /// </summary>
        /// <param name="path">The path to the .sentis model file</param>
        private static Model LoadModel(string path) {
#if !UNITY_STANDALONE
            throw new NotImplementedException();
#endif
            if (!File.Exists(path)) {
                throw new FileNotFoundException($"The .sentis model file was not found at: '{path}'");
            }
            return ModelLoader.Load(path);
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

        /// <summary>
        /// Returns true when this model is found in either StreamingAssets or Resources
        /// </summary>
        public static bool IsModelAvailable(DiffusionModel model) {
            if (ExistsInStreamingAssets(model)) {
                return true;
            }
            if (ExistsInResources(model)) {
                return true;
            }
            return false;
        }

#if UNITY_EDITOR
        /// <summary>
        /// Editor-only check whether a model is in the Resources folder
        /// </summary>
        public static bool ExistsInResources(DiffusionModel model) {
            // check if at least all required files are present in Resources
            return model.All(file => !file.Required || File.Exists(file.ResourcesFilePath));
        }
#endif
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

            var vocab = LoadVocab(model);
            var merges = LoadMerges(model);
            var tokenizerConfig = LoadTokenizerConfig(model);
            var clipTokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var scheduler = Scheduler.FromPretrained(model.File(Path.Combine("scheduler", SchedulerConfig.ConfigName)), backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model.File(Path.Combine("vae_decoder", VaeConfig.ConfigName)), backend);
            var textEncoder = TextEncoder.FromPretrained(model.File(Path.Combine("text_encoder", TextEncoderConfig.ConfigName)), backend);
            var unet = Unet.FromPretrained(model.File(Path.Combine("unet", UnetConfig.ConfigName)), backend);

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

            var vocab = LoadVocab(model);
            var merges = LoadMerges(model);
            var tokenizerConfig = LoadTokenizerConfig(model);
            var tokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            vocab = LoadFromJson<Vocab>(model.Vocab2);
            merges = LoadText(model.Merges2);
            tokenizerConfig = LoadFromJson<TokenizerConfig>(model.TokenizerConfig2);
            var tokenizer2 = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var scheduler = Scheduler.FromPretrained(model.File(Path.Combine("scheduler", SchedulerConfig.ConfigName)), backend);
            //scheduler = Scheduler.FromConfig<PNDMScheduler>(scheduler.Config, backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model.File(Path.Combine("vae_decoder", VaeConfig.ConfigName)), backend);
            var textEncoder = TextEncoder.FromPretrained(model.File(Path.Combine("text_encoder", TextEncoderConfig.ConfigName)), backend);
            var textEncoder2 = TextEncoder.FromPretrained(model.File(Path.Combine("text_encoder_2", TextEncoderConfig.ConfigName)), backend);
            var unet = Unet.FromPretrained(model.File(Path.Combine("unet", UnetConfig.ConfigName)), backend);
 
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