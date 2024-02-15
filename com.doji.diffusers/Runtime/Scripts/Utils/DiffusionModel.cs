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
            $"unet/model.onnx_data",    // optional
            $"unet/weights.pb",         // optional
            $"vae_decoder/model.onnx",
        };

        private const string HF_URL = "https://huggingface.co";

        public string BaseUrl { get { return $"{HF_URL}/{Name}"; } }

        public IEnumerator<(string url, string filePath, bool optional)> GetEnumerator() {
            foreach (string fileName in Files) {
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

        public static readonly DiffusionModel SD_1_5 = new DiffusionModel("runwayml/stable-diffusion-v1-5", "onnx");
        public static readonly DiffusionModel SD_2_1 = new DiffusionModel("julienkay/stable-diffusion-2-1");
        //public static readonly DiffusionModel SD_XL_TURBO = new DiffusionModel("stabilityai/sdxl-turbo");
        //public static readonly DiffusionModel SD_XL_BASE  = new DiffusionModel("stabilityai/stable-diffusion-xl-base-1.0");

        /* all the validated models */
        public static readonly DiffusionModel[] ValidatedModels = new DiffusionModel[] {
                 SD_1_5,
                 SD_2_1
                 //SD_XL_TURBO,
                 //SD_XL_BASE,
        };
    }
}