using System.IO;
using System;
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
        }

        private void OnPromptChanged(string value) {
            Prompt = value;
        }

        private void ExecuteSD() {
            Prompt = PromptField.text;
            Result = _stableDiffusion.RenderTexture;
            _stableDiffusion.Imagine(Prompt, Resolution, Resolution, Steps, GuidanceScale, _negativePrompts);
            Image.texture = Result;
            SaveAs(Result, Prompt);
        }

        private void SaveAs(RenderTexture texture, string prompt) {
            var invalids = Path.GetInvalidFileNameChars();
            var fileName = string.Join("_", prompt.Split(invalids, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
            fileName = fileName[..Math.Min(fileName.Length, 30)];
            string filePath = $"{fileName}_{DateTime.Now:yyyyMMddHHmmss}.png";

            int width = texture.width;
            int height = texture.height;
            Texture2D tex = new Texture2D(width, height, TextureFormat.ARGB32, false);
            RenderTexture.active = texture;
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            File.WriteAllBytes(filePath, tex.EncodeToPNG());
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
    public enum Model { SD_1_5, SD_2_1 }

    public static class ValidatedModelExtensions {
        public static DiffusionModel GetModelInfo(this Model model) {
            return model switch {
                Model.SD_1_5 => DiffusionModel.SD_1_5,
                Model.SD_2_1 => DiffusionModel.SD_2_1,
                _ => throw new System.ArgumentException($"Unknown model '{model}'"),
            };
        }
    }
}
