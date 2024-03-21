using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using static System.IO.Path;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// File information for all components of a diffusion model hosted on
    /// Hugging Face (models, schedulers, and processors) and where to download them.
    /// </summary>
    public class DiffusionModel : IEnumerable<ModelFile> {

        /// <summary>
        /// The string identifying the diffusion model. (e.g. "stabilityai/sdxl-turbo")
        /// </summary>
        public string ModelId { get; private set; }

        /// <summary>
        /// The branch in the HF repository that contains the ONNX files for the diffusion model.
        /// </summary>
        public string Revision { get; private set; }

        /// <summary>
        /// The owner of the repository. (e.g. "stabilityai")
        /// </summary>
        public string Owner { get; private set; }

        /// <summary>
        /// The name of the model. (e.g. "sdxl-turbo")
        /// </summary>
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
            return ModelId.GetHashCode() ^ Revision.GetHashCode();
        }

        public override bool Equals(object obj) {
            DiffusionModel other = (DiffusionModel)obj;
            return ModelId.Equals(other.ModelId) && Revision.Equals(other.Revision);
        }

        public IEnumerable<ModelFile> Files => new List<ModelFile>() {
            ModelIndex,
            SchedulerConfig,
            TextEncoder,
            Merges,
            SpecialTokensMap,
            TokenizerConfig,
            Vocab,
            Unet,
            VaeDecoder,
            VaeEncoder,
        };

        public IEnumerable<ModelFile> OptionalFiles => new List<ModelFile>() {
            TextEncoder2,                                                             // SDXL
            new(this, Combine("text_encoder_2", "model.onnx_data"), required: false), // SDXL
            Merges2,                                                                  // SDXL
            SpecialTokensMap2,                                                        // SDXL
            TokenizerConfig2,                                                         // SDXL
            Vocab2,                                                                   // SDXL
            TextEncoderConfig2,                                                       // SDXL
            new(this, Combine("unet", "model.onnx_data"), required: false),           // stabilityai repo stores external weights that way
            new(this, Combine("unet", "weights.pb"), required: false),                // runwayml repo stores external weights that way
            TextEncoderConfig,                                                        // not present in CompVis/runwayml 1.x onnx models
            UnetConfig,                                                               // not present in CompVis/runwayml 1.x onnx models
            VaeDecoderConfig,                                                         // not present in CompVis/runwayml 1.x onnx models
            VaeEncoderConfig                                                          // not present in CompVis/runwayml 1.x onnx models
        };

        public ModelFile ModelIndex         => new(this, "model_index.json"                               , required: true);
        public ModelFile SchedulerConfig    => new(this, Combine("scheduler", Diffusers.SchedulerConfig.ConfigName), required: true);
        public ModelFile TextEncoder        => new(this, Combine("text_encoder", "model.onnx")            , required: true);
        public ModelFile Merges             => new(this, Combine("tokenizer", "merges.txt")               , required: true);
        public ModelFile SpecialTokensMap   => new(this, Combine("tokenizer", "special_tokens_map.json")  , required: true);
        public ModelFile TokenizerConfig    => new(this, Combine("tokenizer", "tokenizer_config.json")    , required: true);
        public ModelFile Vocab              => new(this, Combine("tokenizer", "vocab.json")               , required: true);
        public ModelFile Unet               => new(this, Combine("unet", "model.onnx")                    , required: true);
        public ModelFile VaeDecoder         => new(this, Combine("vae_decoder, model.onnx")               , required: true);
        public ModelFile VaeEncoder         => new(this, Combine("vae_encoder, model.onnx")               , required: true);
                                                                                                       
        public ModelFile TextEncoder2       => new(this, Combine("text_encoder_2", "model.onnx")          , required: false);
        public ModelFile Merges2            => new(this, Combine("tokenizer_2", "merges.txt")             , required: false);
        public ModelFile SpecialTokensMap2  => new(this, Combine("tokenizer_2", "special_tokens_map.json"), required: false);
        public ModelFile TokenizerConfig2   => new(this, Combine("tokenizer_2", "tokenizer_config.json")  , required: false);
        public ModelFile Vocab2             => new(this, Combine("tokenizer_2", "vocab.json")             , required: false);
        public ModelFile TextEncoderConfig  => new(this, Combine("text_encoder", Diffusers.TextEncoderConfig.ConfigName), required: false);
        public ModelFile TextEncoderConfig2 => new(this, Combine("text_encoder_2", Diffusers.TextEncoderConfig.ConfigName), required: false);
        public ModelFile UnetConfig         => new(this, Combine("unet", Diffusers.UnetConfig.ConfigName) , required: false);
        public ModelFile VaeDecoderConfig   => new(this, Combine("vae_decoder", VaeConfig.ConfigName)     , required: false);
        public ModelFile VaeEncoderConfig   => new(this, Combine("vae_encoder", VaeConfig.ConfigName)     , required: false);

        public ModelFile File(string path, bool required = true) {
            return new ModelFile(this, path, required);
        }
        
        internal const string HF_URL = "https://huggingface.co";

        public string BaseUrl { get { return $"{HF_URL}/{ModelId}"; } }

        public IEnumerator<ModelFile> GetEnumerator() {
            foreach (var file in Files) {
                yield return file;
            }
            foreach (var file in OptionalFiles) {
                yield return file;
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
        public readonly string Url => $"{Model.BaseUrl}/resolve/{Model.Revision}/{FileName}";

#if UNITY_EDITOR
        /// <summary>
        /// The path to the Resources folder location of this file.
        /// E.g. "Packages/com.doji.diffusers/Runtime/Resources/stabilityai/sdxl-turbo/main/unet/model.onnx"
        /// Use this in-Editor only, to check whether the file is present.
        /// To actually load the file from Resources use <see cref="ResourcePath"/>
        /// </summary>
        public readonly string ResourcesFilePath
            => Path.Combine("Packages", "com.doji.diffusers", "Runtime", "Resources", Model.Owner, Model.ModelName, Model.Revision, FileName);
#endif

        /// <summary>
        /// The path to this file when loading from Resources.
        /// E.g. "stabilityai/sdxl-turbo/main/unet/model"
        /// (local to Resources, without file extension)
        /// </summary>
        public readonly string ResourcePath
            => Path.Combine(Model.Owner, Model.ModelName, Model.Revision, Path.ChangeExtension(FileName, null));

        /// <summary>
        /// The path to the StreamingAssets location of this file.
        /// </summary>
        public readonly string StreamingAssetsPath {
            get {
                string ext = Path.GetExtension(FileName);
                string fileName = FileName;
                if (ext == ".onnx") {
                    fileName = Path.ChangeExtension(fileName, ".sentis"); // .onnx -> .sentis
                }
                return Path.Combine(UnityEngine.Application.streamingAssetsPath, Model.Owner, Model.ModelName, Model.Revision, fileName);
            }
        }
           
        /// <summary>
        /// Creates a ModelFile pointing to the given file on this same model.
        /// </summary>
        public ModelFile New(string fileName) {
            string directory = Path.GetDirectoryName(FileName);
            string newFilePath = Path.Combine(directory, fileName);
            return new ModelFile(Model, newFilePath, Required);
        }

        /// <summary>
        /// Whether all diffusion models will include this file.
        /// Used to determine whether to log errors in case the file can not be found.
        /// </summary>
        public bool Required;

        public readonly DiffusionModel Model;
        public readonly string FileName;

        public ModelFile(DiffusionModel model, string fileName, bool required) {
            Model = model;
            FileName = fileName;
            Required = required;
        }
    }
}