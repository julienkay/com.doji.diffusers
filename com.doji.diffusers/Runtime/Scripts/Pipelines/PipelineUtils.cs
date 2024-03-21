using Doji.AI.Transformers;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Contains logic related to pipeline loading
    /// </summary>
    public static class PipelineUtils {

        /// <summary>
        /// Turns this pipeline into another pipeline of type <typeparamref name="T"/> reusing its components where applicable. 
        /// </summary>
        public static T As<T>(this DiffusionPipeline pipe) where T : IDiffusionPipeline {
            if (pipe is T p) {
                return p;
            }
            if (typeof(T) == typeof(ITxt2ImgPipeline)) {
                return pipe switch {
                    StableDiffusionImg2ImgPipeline => (T)(new StableDiffusionPipeline(pipe) as ITxt2ImgPipeline),
                    StableDiffusionXLImg2ImgPipeline => (T)(new StableDiffusionXLPipeline(pipe) as ITxt2ImgPipeline),
                    _ => throw new NotImplementedException($"Conversion to {typeof(T).Name} not yet implemented for {pipe.GetType().Name}")
                };
            } else if (typeof(T) == typeof(IImg2ImgPipeline)) {
                return pipe switch {
                    StableDiffusionPipeline => (T)(new StableDiffusionImg2ImgPipeline(pipe) as IImg2ImgPipeline),
                    StableDiffusionXLPipeline => (T)(new StableDiffusionXLImg2ImgPipeline(pipe) as IImg2ImgPipeline),
                    _ => throw new NotImplementedException($"Conversion to {typeof(T).Name} not yet implemented for {pipe.GetType().Name}")
                };
            } else {
                throw new ArgumentException($"Unknown pipeline interface {typeof(T).Name}");
            }
        }
    }

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

        /// <summary>
        /// Loads the given model into a txt2img pipeline.
        /// </summary>
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
            // check if at least all required files are present in StreamingAssets
            return model.All(file => !file.Required || File.Exists(file.StreamingAssetsPath));
        }
    }

    public partial class StableDiffusionPipeline {

        public static new StableDiffusionPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
            PipelineConfig config = LoadPipelineConfig(model);

            var vocab = LoadVocab(model);
            var merges = LoadMerges(model);
            var tokenizerConfig = LoadTokenizerConfig(model);
            var clipTokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var scheduler = Scheduler.FromPretrained(model, backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model, backend);
            var textEncoder = TextEncoder.FromPretrained(model.TextEncoder, model.TextEncoderConfig, backend);
            var unet = Unet.FromPretrained(model, backend);

            StableDiffusionPipeline sdPipeline = new StableDiffusionPipeline(
                vaeDecoder,
                textEncoder,
                clipTokenizer,
                scheduler,
                unet,
                backend
            );
            sdPipeline.ModelInfo = model;
            sdPipeline.Config = config;
            return sdPipeline;
        }

        /// <summary>
        /// Create a txt2img pipeline reusing the components of the given pipeline.
        /// </summary>
        public StableDiffusionPipeline(DiffusionPipeline pipe) : base(pipe._ops.backendType) {
            if (pipe is not StableDiffusionImg2ImgPipeline) {
                throw new InvalidOperationException($"Cannot create StableDiffusionPipeline from a {pipe.GetType().Name}.");
            }
            ModelInfo = pipe.ModelInfo;
            Config = pipe.Config;
            VaeDecoder = pipe.VaeDecoder;
            Tokenizer = pipe.Tokenizer;
            TextEncoder = pipe.TextEncoder;
            Scheduler = pipe.Scheduler;
            Unet = pipe.Unet;
        }
    }

    public partial class StableDiffusionImg2ImgPipeline {

        public static new StableDiffusionImg2ImgPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
            PipelineConfig config = LoadPipelineConfig(model);

            var vocab = LoadVocab(model);
            var merges = LoadMerges(model);
            var tokenizerConfig = LoadTokenizerConfig(model);
            var clipTokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            var scheduler = Scheduler.FromPretrained(model, backend);
            var vaeEncoder = VaeEncoder.FromPretrained(model, backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model, backend);
            var textEncoder = TextEncoder.FromPretrained(model.TextEncoder, model.TextEncoderConfig, backend);
            var unet = Unet.FromPretrained(model, backend);

            StableDiffusionImg2ImgPipeline sdPipeline = new StableDiffusionImg2ImgPipeline(
                vaeEncoder,
                vaeDecoder,
                textEncoder,
                clipTokenizer,
                scheduler,
                unet,
                backend
            );
            sdPipeline.ModelInfo = model;
            sdPipeline.Config = config;
            return sdPipeline;
        }

        /// <summary>
        /// Create an img2img pipeline reusing the components of the given pipeline.
        /// </summary>
        public StableDiffusionImg2ImgPipeline(DiffusionPipeline pipe) : base(pipe._ops.backendType) {
            ModelInfo = pipe.ModelInfo;
            Config = pipe.Config;

            if (pipe is StableDiffusionPipeline) {
                VaeEncoder = VaeEncoder.FromPretrained(pipe.ModelInfo, pipe._ops.backendType);
            } else {
                throw new InvalidOperationException($"Cannot create StableDiffusionImg2ImgPipeline from a {pipe.GetType().Name}.");
            }

            ImageProcessor = new VaeImageProcessor(/*vaeScaleFactor: self.vae_scale_factor*/);
            VaeDecoder = pipe.VaeDecoder;
            TextEncoder = pipe.TextEncoder;
            Tokenizer = pipe.Tokenizer;
            Scheduler = pipe.Scheduler;
            Unet = pipe.Unet;
        }
    }

    public partial class StableDiffusionXLPipeline {

        public static new StableDiffusionXLPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
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
            var scheduler = Scheduler.FromPretrained(model, backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model, backend);
            var textEncoder = TextEncoder.FromPretrained(model.TextEncoder, model.TextEncoderConfig, backend);
            var textEncoder2 = TextEncoder.FromPretrained(model.TextEncoder2, model.TextEncoderConfig2, backend);
            var unet = Unet.FromPretrained(model, backend);

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
            sdPipeline.ModelInfo = model;
            sdPipeline.Config = config;
            return sdPipeline;
        }

        /// <summary>
        /// Create an SDXL img2img pipeline reusing the components of the given pipeline.
        /// </summary>
        public StableDiffusionXLPipeline(DiffusionPipeline pipe) : base(pipe._ops.backendType) {
            ModelInfo = pipe.ModelInfo;
            Config = pipe.Config;

            if (pipe is StableDiffusionXLImg2ImgPipeline xl) {
                Tokenizer2 = xl.Tokenizer2;
                TextEncoder2 = xl.TextEncoder2;
                VaeScaleFactor = xl.VaeScaleFactor;
                Encoders = xl.Encoders;
            } else {
                throw new InvalidOperationException($"Cannot create StableDiffusionXLImg2ImgPipeline from a {pipe.GetType().Name}.");
            }

            VaeDecoder = pipe.VaeDecoder;
            TextEncoder = pipe.TextEncoder;
            Tokenizer = pipe.Tokenizer;
            Scheduler = pipe.Scheduler;
            Unet = pipe.Unet;
        }
    }

    public partial class StableDiffusionXLImg2ImgPipeline {

        public static new StableDiffusionXLImg2ImgPipeline FromPretrained(DiffusionModel model, BackendType backend = BackendType.GPUCompute) {
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
            var scheduler = Scheduler.FromPretrained(model, backend);
            var vaeEncoder = VaeEncoder.FromPretrained(model, backend);
            var vaeDecoder = VaeDecoder.FromPretrained(model, backend);
            var textEncoder = TextEncoder.FromPretrained(model.TextEncoder, model.TextEncoderConfig, backend);
            var textEncoder2 = TextEncoder.FromPretrained(model.TextEncoder2, model.TextEncoderConfig2, backend);
            var unet = Unet.FromPretrained(model, backend);

            StableDiffusionXLImg2ImgPipeline sdPipeline = new StableDiffusionXLImg2ImgPipeline(
                vaeEncoder,
                vaeDecoder,
                textEncoder,
                tokenizer,
                scheduler,
                unet,
                textEncoder2,
                tokenizer2,
                backend
            );
            sdPipeline.ModelInfo = model;
            sdPipeline.Config = config;
            return sdPipeline;
        }

        /// <summary>
        /// Create an SDXL img2img pipeline reusing the components of the given pipeline.
        /// </summary>
        public StableDiffusionXLImg2ImgPipeline(DiffusionPipeline pipe) : base(pipe._ops.backendType) {
            ModelInfo = pipe.ModelInfo;
            Config = pipe.Config;

            if (pipe is StableDiffusionXLPipeline xl) {
                VaeEncoder = VaeEncoder.FromPretrained(pipe.ModelInfo, pipe._ops.backendType);
                Tokenizer2 = xl.Tokenizer2;
                TextEncoder2 = xl.TextEncoder2;
                VaeScaleFactor = xl.VaeScaleFactor;
                Encoders = xl.Encoders;
            } else {
                throw new InvalidOperationException($"Cannot create StableDiffusionXLImg2ImgPipeline from a {pipe.GetType().Name}.");
            }

            ImageProcessor = new VaeImageProcessor(/*vaeScaleFactor: self.vae_scale_factor*/);
            VaeDecoder = pipe.VaeDecoder;
            TextEncoder = pipe.TextEncoder;
            Tokenizer = pipe.Tokenizer;
            Scheduler = pipe.Scheduler;
            Unet = pipe.Unet;
        }
    }
}
