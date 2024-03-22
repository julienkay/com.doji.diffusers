using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Doji.AI.Diffusers.Samples {

    public class StableDiffusionUI : MonoBehaviour {

        public TMP_Dropdown ModelDropdown;
        public RawImage Image;

        public Model Model = Model.SD_1_5;
        public int Resolution = 512;
        public int Steps = 20;
        public float GuidanceScale = 7.5f;
        public Texture2D InputImage;

        public RenderTexture Result;
        private StableDiffusion _stableDiffusion;

        private const string _negativePrompts = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting";

        private void Start() {
            _stableDiffusion = new StableDiffusion(Model.GetModelInfo());

            List<TMP_Dropdown.OptionData> options = new List<TMP_Dropdown.OptionData>();
            for (int i = 0; i < Enum.GetNames(typeof(Model)).Length; i++) {
                options.Add(new TMP_Dropdown.OptionData(Enum.GetName(typeof(Model), i)));
            }
            ModelDropdown.options = options;
            ModelDropdown.onValueChanged.AddListener(OnModelChanged);
            ModelDropdown.value = (int)Model; 
        }

        private void OnModelChanged(int modelIndex) {
            var model = ((Model)modelIndex).GetModelInfo();
            if (_stableDiffusion.Model != model) {
                _stableDiffusion.Model = model;
            }
        }

        public void Txt2Img(string prompt) {
            Result = _stableDiffusion.Result;
            Metadata m = _stableDiffusion.Imagine(prompt, Resolution, Resolution, Steps, GuidanceScale, _negativePrompts);
            Image.texture = Result;
            PNGUtils.SaveToDisk(Result, ".", m);
        }

        public async Task Txt2ImgAsync(string prompt) {
            Result = _stableDiffusion.Result;
            Metadata m = await _stableDiffusion.ImagineAsync(prompt, Resolution, Resolution, Steps, GuidanceScale, _negativePrompts);
            Image.texture = Result;
            PNGUtils.SaveToDisk(Result, ".", m);
        }

        public void Img2Img(string prompt, float strength) {
            Result = _stableDiffusion.Result;
            Metadata m = _stableDiffusion.Imagine(prompt, InputImage, Steps, GuidanceScale, _negativePrompts, strength: strength);
            Image.texture = Result;
            PNGUtils.SaveToDisk(Result, ".", m);
        }

        private void OnDestroy() {
            _stableDiffusion?.Dispose();
        }

#if UNITY_EDITOR
        private void OnValidate() {
            if (_stableDiffusion == null)
                return;

            DiffusionModel model = Model.GetModelInfo();
            if (_stableDiffusion.Model != model) {
                ModelDropdown.value = (int)Model;
            }
        }
#endif
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
                _ => throw new ArgumentException($"Unknown model '{model}'"),
            };
        }
    }
}
