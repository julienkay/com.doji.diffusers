using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// This class provides access to <see cref="StableDiffusionPipeline"/> objects.
    /// TODO: For easy access, use StableDiffusion.FromPretrained(...)
    /// </summary>
    public class StableDiffusion : IDisposable {

        public BackendType Backend {
            get => _backend;
            set {
                if (_backend != value) {
                    Dispose();
                    _backend = value;
                    Initialize();
                }
            }
        }
        private BackendType _backend = BackendType.GPUCompute;


        private StableDiffusionPipeline _sdPipeline;

        public RenderTexture RenderTexture;

        public StableDiffusion() {
            Initialize();
        }

        private void Initialize() {
            _sdPipeline = StableDiffusionPipeline.FromPretrained(DiffusionModel.SD_1_5, Backend);
            RenderTexture = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32);
        }

        public void Imagine(string prompt, int width, int height, int numInferenceSteps = 50, float guidanceScale = 7.5f) {
            RenderTexture.name = prompt;
            var image = _sdPipeline.Generate(
                prompt,
                width: width,
                height: height,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale
            );
            TextureConverter.RenderToTexture(image, RenderTexture);
        }

        public void Dispose() {
            _sdPipeline?.Dispose();
            if (RenderTexture != null ) {
                RenderTexture.Release();
            }
        }
    }
}
