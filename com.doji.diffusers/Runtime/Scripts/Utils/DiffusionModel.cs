using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stores file information for all components of a diffusion model.
    /// (models, schedulers, and processors) and where to download them.
    /// </summary>
    public class DiffusionModel : IEnumerable<(string url, string filePath, bool optional)> {

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

        private const string ModelIndex = "model_index.json";

        public static readonly IEnumerable<string> Files = new List<string>() {
            ModelIndex,
            $"scheduler/scheduler_config.json",
            $"text_encoder/model.onnx",
            $"tokenizer/merges.txt",
            $"tokenizer/special_tokens_map.json",
            $"tokenizer/tokenizer_config.json",
            $"tokenizer/vocab.json",
            $"unet/model.onnx",
            $"vae_decoder/model.onnx",
            $"vae_encoder/model.onnx"
        };

        public static readonly IEnumerable<string> OptionalFiles = new List<string>() {
            $"text_encoder_2/model.onnx",           // SDXL
            $"tokenizer_2/merges.txt",              // SDXL
            $"tokenizer_2/special_tokens_map.json", // SDXL
            $"tokenizer_2/tokenizer_config.json",   // SDXL
            $"tokenizer_2/vocab.json",              // SDXL
            $"unet/model.onnx_data",                // runwayml repo stores it that way
            $"unet/weights.pb"                      // runwayml repo stores it that way
        };

        private const string HF_URL = "https://huggingface.co";

        public string BaseUrl { get { return $"{HF_URL}/{Name}"; } }

        public IEnumerator<(string url, string filePath, bool optional)> GetEnumerator() {
            foreach (string fileName in Files) {
                string url = $"{HF_URL}/{Name}/resolve/{Revision}/{fileName}";
                string filePath = Path.Combine("Packages", "com.doji.diffusers", "Runtime", "Resources", Name, $"{fileName}");
                yield return (url, filePath, false);
            }
            foreach (string fileName in OptionalFiles) {
                string url = $"{HF_URL}/{Name}/resolve/{Revision}/{fileName}";
                string filePath = Path.Combine("Packages", "com.doji.diffusers", "Runtime", "Resources", Name, $"{fileName}");
                yield return (url, filePath, true);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }

        public static readonly DiffusionModel SD_1_5 = new DiffusionModel("runwayml/stable-diffusion-v1-5", "onnx");
        public static readonly DiffusionModel SD_2_1 = new DiffusionModel("julienkay/stable-diffusion-2-1");
        public static readonly DiffusionModel SD_XL_BASE  = new DiffusionModel("stabilityai/stable-diffusion-xl-base-1.0");
        //public static readonly DiffusionModel SD_XL_TURBO = new DiffusionModel("stabilityai/sdxl-turbo");

        /* all the validated models */
        public static readonly DiffusionModel[] ValidatedModels = new DiffusionModel[] {
                 SD_1_5,
                 SD_2_1,
                 SD_XL_BASE
                 //SD_XL_TURBO,
        };
    }
}