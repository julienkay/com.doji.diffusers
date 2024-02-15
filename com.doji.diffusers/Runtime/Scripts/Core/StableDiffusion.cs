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

        public StableDiffusion(DiffusionModel model) {
            _sdPipeline = StableDiffusionPipeline.FromPretrained(model, Backend);
            RenderTexture = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32);
        }

        private void Initialize() {
           
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
