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

        public RenderTexture Result;
        private StableDiffusion sd;
        public string Prompt = "a cat";
        public int Resolution = 512;
        public int Steps = 20;
        public float GuidanceScale = 7.5f;

        private void Start() {
            sd = new StableDiffusion();
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
            Result = sd.RenderTexture;
            sd.Imagine(Prompt, Resolution, Resolution, Steps, GuidanceScale);
            Image.texture = Result;
            //SaveAs(Result, Prompt);
        }

        private void SaveAs(RenderTexture texture, string prompt) {
            var invalids = Path.GetInvalidFileNameChars();
            var fileName = string.Join("_", prompt.Split(invalids, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
            string filePath = $"{fileName}_{DateTime.Now:yyyyMMddHHmmss}.png";

            int width = texture.width;
            int height = texture.height;
            Texture2D tex = new Texture2D(width, height, TextureFormat.ARGB32, false);
            RenderTexture.active = texture;
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            File.WriteAllBytes(filePath, tex.EncodeToPNG());
        }

        private void OnDestroy() {
            sd?.Dispose();
        }
    }
}
