using Doji.AI.Transformers;
using Newtonsoft.Json;
using System.IO;
using Unity.Sentis;
using UnityEngine;

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
    }
}