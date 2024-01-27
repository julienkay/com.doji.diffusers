using Doji.AI.Transformers;
using Newtonsoft.Json;
using System.IO;
using Unity.Sentis;
using UnityEngine;
using System.Collections.Generic;
using System.Collections;

namespace Doji.AI.Diffusers {

    public partial class StableDiffusionPipeline {

        internal static Model LoadVaeDecoder() {
            string path = Path.Combine("vae_decoder", "model");
            return LoadFromModelAsset(path);
        }

        internal static Model LoadTextEncoder() {
            string path = Path.Combine("text_encoder", "model");
            return LoadFromModelAsset(path);
        }

        internal static Vocab LoadVocab() {
            string path = Path.Combine("tokenizer", "vocab");
            TextAsset vocabFile = Resources.Load<TextAsset>(path);
            var vocab = Vocab.Deserialize(vocabFile.text);
            Resources.UnloadAsset(vocabFile);
            return vocab;
        }

        internal static string LoadMerges() {
            string path = Path.Combine("tokenizer", "merges");
            return LoadFromTextAsset(path);
        }

        internal static TokenizerConfig LoadTokenizerConfig() {
            string path = Path.Combine("tokenizer", "tokenizer_config");
            return LoadJsonFromTextAsset<TokenizerConfig>(path);
        }

        internal static Model LoadUnet() {
            string path = Path.Combine("unet", "model");
            return LoadFromModelAsset(path);
        }

        private static string LoadFromTextAsset(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path)
                ?? throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            string text = textAsset.text;
            Resources.UnloadAsset(textAsset);
            return text;
        }

        private static T LoadJsonFromTextAsset<T>(string path) {
            TextAsset textAsset = Resources.Load<TextAsset>(path)
                ?? throw new FileNotFoundException($"The TextAsset file was not found at: '{path}'");
            T deserializedObject = JsonConvert.DeserializeObject<T>(textAsset.text);
            Resources.UnloadAsset(textAsset);
            return deserializedObject;
        }

        private static Model LoadFromModelAsset(string path) {
            ModelAsset modelAsset = Resources.Load<ModelAsset>(path)
                ?? throw new FileNotFoundException($"The ModelAsset file was not found at: '{path}'");
            Model model = ModelLoader.Load(modelAsset);
            Resources.UnloadAsset(modelAsset);
            return model;
        }

        public static StableDiffusionPipeline FromPretrained(BackendType backend = BackendType.GPUCompute) {
            var vocab = LoadVocab();
            var merges = LoadMerges();
            var tokenizerConfig = LoadTokenizerConfig();
            var clipTokenizer = new ClipTokenizer(
                vocab,
                merges,
                tokenizerConfig
            );
            //FIXME: Load from scheduler config
            var scheduler = new PNDMScheduler(
                  betaEnd: 0.012f,
                  betaSchedule: Schedule.ScaledLinear,
                  betaStart: 0.00085f,
                  numTrainTimesteps: 1000,
                  setAlphaToOne: false,
                  skipPrkSteps: true,
                  stepsOffset: 1,
                  trainedBetas: null
            );
            var vaeDecoder = LoadVaeDecoder();
            var textEncoder = LoadTextEncoder();
            var unet = LoadUnet();
            StableDiffusionPipeline sdPipeline = new StableDiffusionPipeline(
                vaeDecoder,
                textEncoder,
                clipTokenizer,
                scheduler,
                unet,
                backend
            );
            return sdPipeline;
        }
        
        public struct DiffusionModel : IEnumerable<(string url, string filePath, bool optional)> {

            private const string HF_URL = "https://huggingface.co";

            public string Name { get; private set; }
            public string Revision { get; private set; }

            public DiffusionModel(string modelName, string revision = null) {
                Name = modelName;
                Revision = revision ?? "main";
            }

            public override int GetHashCode() {
                return Name.GetHashCode();
            }

            public override bool Equals(object obj) {
                DiffusionModel other = (DiffusionModel)obj;
                return Name.Equals(other.Name);
            }

            public static readonly IEnumerable<string> FileNames = new List<string>() {
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/model.onnx",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "unet/model.onnx",
                "unet/model.onnx_data",    // optional
                "unet/weights.pb",         // optional
                "vae_decoder/model.onnx",
            };

            public IEnumerator<(string url, string filePath, bool optional)> GetEnumerator() {
                foreach (string fileName in FileNames) {
                    bool optional = false;
                    if (fileName == "unet/model.onnx_data" || fileName == "unet/weights.pb") {
                        optional = true;  
                    }
                    string url = $"{HF_URL}/{Name}/resolve/{Revision}/{fileName}";
                    string filePath = Path.Combine("Packages", "com.doji.diffusers", "Runtime", "Resources", Name, $"{fileName}");
                    yield return (url, filePath, optional);
                }
            }

            IEnumerator IEnumerable.GetEnumerator() {
                return GetEnumerator();
            }

            public static readonly DiffusionModel SD_1_5      = new DiffusionModel("runwayml/stable-diffusion-v1-5", "onnx");
            public static readonly DiffusionModel SD_XL_TURBO = new DiffusionModel("stabilityai/sdxl-turbo");
            public static readonly DiffusionModel SD_XL_BASE  = new DiffusionModel("stabilityai/stable-diffusion-xl-base-1.0");
            public static readonly DiffusionModel SD_2_1      = new DiffusionModel("aislamov/stable-diffusion-2-1-base-onnx");

            /* all the validated models */
            public static readonly DiffusionModel[] ValidatedModels = new DiffusionModel[] {
                 SD_1_5,
                 //SD_XL_TURBO,
                 //SD_XL_BASE,
                 //SD_2_1
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