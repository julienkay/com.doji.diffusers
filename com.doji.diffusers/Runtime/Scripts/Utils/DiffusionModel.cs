using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// File information for all components of a diffusion model hosted on
    /// Hugging Face (models, schedulers, and processors) and where to download them.
    /// </summary>
    public class DiffusionModel : IEnumerable<ModelFile> {

        public string ModelId { get; private set; }
        public string Revision { get; private set; }
        public string Owner { get; private set; }
        public string ModelName { get; private set; }

        public DiffusionModel(string modelId, string revision = "main") {
            string[] parts = modelId.Split("/");
            if (parts.Length != 2) {
                throw new ArgumentException("ModelId has invalid format", nameof(modelId));
            }
            ModelId = modelId;
            Revision = revision;
            Owner = parts[0];
            ModelName = parts[1];
        }

        public override int GetHashCode() {
            return ModelId.GetHashCode();
        }

        public override bool Equals(object obj) {
            DiffusionModel other = (DiffusionModel)obj;
            return ModelId.Equals(other.ModelId);
        }

        public static readonly IEnumerable<string> Files = new List<string>() {
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/model.onnx",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "unet/model.onnx",
            "vae_decoder/model.onnx",
            "vae_encoder/model.onnx"
        };

        public static readonly IEnumerable<string> OptionalFiles = new List<string>() {
            "text_encoder_2/model.onnx",           // SDXL
            "text_encoder_2/model.onnx_data",      // SDXL
            "tokenizer_2/merges.txt",              // SDXL
            "tokenizer_2/special_tokens_map.json", // SDXL
            "tokenizer_2/tokenizer_config.json",   // SDXL
            "tokenizer_2/vocab.json",              // SDXL
            "unet/model.onnx_data",                // runwayml repo stores it that way
            "unet/weights.pb",                     // runwayml repo stores it that way
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "text_encoder_2/config.json",
            "unet/config.json",
            "vae_decoder/config.json",
            "vae_encoder/config.json"
        };

        internal const string HF_URL = "https://huggingface.co";

        public string BaseUrl { get { return $"{HF_URL}/{ModelId}"; } }

        public IEnumerator<ModelFile> GetEnumerator() {
            foreach (string fileName in Files) {
                yield return new ModelFile(this, fileName, required: true);
            }
            foreach (string fileName in OptionalFiles) {
                yield return new ModelFile(this, fileName, required: false);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }

        public static readonly DiffusionModel SD_1_5 = new DiffusionModel("runwayml/stable-diffusion-v1-5", "onnx");
        public static readonly DiffusionModel SD_2_1 = new DiffusionModel("julienkay/stable-diffusion-2-1");
        public static readonly DiffusionModel SD_XL_BASE  = new DiffusionModel("stabilityai/stable-diffusion-xl-base-1.0");
        public static readonly DiffusionModel SD_TURBO  = new DiffusionModel("julienkay/sd-turbo");
        public static readonly DiffusionModel SD_XL_TURBO = new DiffusionModel("stabilityai/sdxl-turbo");

        /* all the validated models */
        public static readonly DiffusionModel[] ValidatedModels = new DiffusionModel[] {
                 SD_1_5,
                 SD_2_1,
                 SD_XL_BASE,
                 SD_TURBO,
                 SD_XL_TURBO,
        };
    }

    public struct ModelFile {

        /// <summary>
        /// The source url on HF.
        /// </summary>
        public readonly string Url => $"{DiffusionModel.HF_URL}/{_model.ModelId}/resolve/{_model.Revision}/{_fileName}";

        /// <summary>
        /// The path to the Resources folder location of this file.
        /// E.g. "Packages/com.doji.diffusers/Runtime/Resources/stabilityai/stable-diffusion-2-1/main/unet/model.onnx"
        /// To actually load the file from Resources use <see cref="ResourcePath"/>
        /// </summary>
        public readonly string ResourcesFilePath
            => Path.Combine("Packages", "com.doji.diffusers", "Runtime", "Resources", _model.Owner, _model.ModelName, _model.Revision, _fileName);

        /// <summary>
        /// The path to load this file from when loading from Resources.
        /// E.g. "stabilityai/stable-diffusion-2-1/main/unet/model"
        /// (local to Resources, without file extension)
        /// </summary>
        public readonly string ResourcePath
            => Path.Combine(_model.Owner, _model.ModelName, _model.Revision, Path.ChangeExtension(_fileName, null));

        /// <summary>
        /// The path to the StreamingAssets location of this file.
        /// </summary>
        public readonly string StreamingAssetsPath {
            get {
                string ext = Path.GetExtension(_fileName);
                string fileName = _fileName;
                if (ext == ".onnx") {
                    fileName = Path.ChangeExtension(fileName, ".sentis"); // .onnx -> .sentis
                }
                return Path.Combine(UnityEngine.Application.streamingAssetsPath, _model.Owner, _model.ModelName, _model.Revision, fileName);
            }
        }
           
        /// <summary>
        /// Whether all diffusion models will include this file.
        /// Used to determine whether to log errors in case the file can not be found.
        /// </summary>
        public bool Required;

        private DiffusionModel _model;
        private string _fileName;

        public ModelFile(DiffusionModel model, string fileName, bool required) {
            _model = model;
            _fileName = fileName;
            Required = required;
        }
    }
}