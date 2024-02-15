using System;
using System.Threading.Tasks;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// This class provides access to <see cref="StableDiffusionPipeline"/> objects.
    /// </summary>
    public class StableDiffusion : IDisposable {

        public DiffusionModel Model {
            get => _model;
            set {
                if (_model != value) {
                    _sdPipeline.Dispose();
                    _model = value;
                    Initialize();
                }
            }
        }
        private DiffusionModel _model;

        public BackendType Backend {
            get => _backend;
            set {
                if (_backend != value) {
                    _sdPipeline.Dispose();
                    _backend = value;
                    Initialize();
                }
            }
        }
        private BackendType _backend = BackendType.GPUCompute;

        private StableDiffusionPipeline _sdPipeline;
        public RenderTexture RenderTexture;

        public StableDiffusion(DiffusionModel model) {
            _model = model;
            Initialize();
        }

        private void Initialize() {
            _sdPipeline = StableDiffusionPipeline.FromPretrained(Model, Backend);
            if (RenderTexture == null) {
                RenderTexture = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32);
            }
        }

        public void Imagine(string prompt, int width, int height, int numInferenceSteps = 50, float guidanceScale = 7.5f, string negativePrompt = null) {
            RenderTexture.name = prompt;
            var image = _sdPipeline.Generate(
                prompt,
                width: width,
                height: height,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                negativePrompt: negativePrompt
            );
            Profiler.BeginSample("Convert to RenderTexture");
            TextureConverter.RenderToTexture(image, RenderTexture);
            Profiler.EndSample();
        }

        public async Task ImagineAsync(string prompt, int width, int height, int numInferenceSteps = 50, float guidanceScale = 7.5f, string negativePrompt = null) {
            RenderTexture.name = prompt;
            var image = _sdPipeline.Generate(
                prompt,
                width: width,
                height: height,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                negativePrompt: negativePrompt
            );
            await image.ReadbackRequestAsync();
            Profiler.BeginSample("Convert to rendertexture");
            TextureConverter.RenderToTexture(image, RenderTexture);
            Profiler.EndSample();
        }

        public void Dispose() {
            _sdPipeline?.Dispose();
            if (RenderTexture != null) {
                RenderTexture.Release();
            }
        }
    }
}
