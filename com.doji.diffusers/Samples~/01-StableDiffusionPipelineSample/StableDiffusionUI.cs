using System;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Doji.AI.Diffusers.Samples {

    public class StableDiffusionUI : MonoBehaviour {

        public RawImage Image;
        public TMP_InputField PromptField;
        public Button GenerateButton;

        public Model Model = Model.SD_1_5;
        public string Prompt = "a cat";
        public int Resolution = 512;
        public int Steps = 20;
        public float GuidanceScale = 7.5f;
        
        public RenderTexture Result;
        private StableDiffusion _stableDiffusion;

        private const string _negativePrompts = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting";

        private void Start() {
            _stableDiffusion = new StableDiffusion(Model.GetModelInfo());
            PromptField.onValueChanged.AddListener(OnPromptChanged);
            GenerateButton.onClick.AddListener(OnGenerateClicked);
        }

        private void OnGenerateClicked() {
            ExecuteSD();
            /*try {
                ExecuteSDAsync();
            } catch (Exception ex) {
                throw ex;
            }*/
        }

        private void OnPromptChanged(string value) {
            Prompt = value;
        }

        private void ExecuteSD() {
            Prompt = PromptField.text;
            Result = _stableDiffusion.RenderTexture;
            Parameters p = _stableDiffusion.Imagine(Prompt, Resolution, Resolution, Steps, GuidanceScale, _negativePrompts);
            Image.texture = Result;
            PNGUtils.SaveToDisk(Result, ".", p);
        }

        private async Task ExecuteSDAsync() {
            Prompt = PromptField.text;
            Result = _stableDiffusion.RenderTexture;
            Parameters p = await _stableDiffusion.ImagineAsync(Prompt, Resolution, Resolution, Steps, GuidanceScale, _negativePrompts);
            Image.texture = Result;
            PNGUtils.SaveToDisk(Result, ".", p);
        }

        private void OnDestroy() {
            _stableDiffusion?.Dispose();
        }

        private void OnValidate() {
            if (_stableDiffusion == null)
                return;

            DiffusionModel model = Model.GetModelInfo();
            if (_stableDiffusion.Model != model) {
                _stableDiffusion.Model = model;
            }
        }
    }

    /// <summary>
    /// Enum whose values map to certain known diffusion models (<see cref="DiffusionModel.ValidatedModels"/>)
    /// </summary>
    public enum Model { SD_1_5, SD_2_1, SD_XL, SD_TURBO, SD_XL_TURBO }

    public static class ValidatedModelExtensions {
        public static DiffusionModel GetModelInfo(this Model model) {
            return model switch {
                Model.SD_1_5 => DiffusionModel.SD_1_5,
                Model.SD_2_1 => DiffusionModel.SD_2_1,
                Model.SD_XL => DiffusionModel.SD_XL_BASE,
                Model.SD_TURBO => DiffusionModel.SD_TURBO,
                Model.SD_XL_TURBO => DiffusionModel.SD_XL_TURBO,
                _ => throw new System.ArgumentException($"Unknown model '{model}'"),
            };
        }
    }
}
