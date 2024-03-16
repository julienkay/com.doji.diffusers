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

        private DiffusionPipeline _sdPipeline;
        private DiffusionPipelineAsync _asyncPipeline;
        public RenderTexture RenderTexture;

        public StableDiffusion(DiffusionModel model) {
            _model = model;
            Initialize();
        }

        private void Initialize() {
            _sdPipeline = DiffusionPipeline.FromPretrained(Model, Backend);
            _asyncPipeline = (DiffusionPipelineAsync)_sdPipeline;
            if (RenderTexture == null) {
                RenderTexture = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32);
            }
        }

        public Parameters Imagine(string prompt, int width, int height, int numInferenceSteps = 50, float guidanceScale = 7.5f, string negativePrompt = null) {
            RenderTexture.name = prompt;

            Profiler.BeginSample("StableDiffusion.Imagine");
            var image = _sdPipeline.Generate(
                prompt,
                width: width,
                height: height,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                negativePrompt: negativePrompt
            );
            Profiler.EndSample();

            Profiler.BeginSample("Convert to RenderTexture");
            TextureConverter.RenderToTexture(image, RenderTexture);
            Profiler.EndSample();

            return _sdPipeline.GetParameters();
        }

        public async Task<Parameters> ImagineAsync(string prompt, int width, int height, int numInferenceSteps = 50, float guidanceScale = 7.5f, string negativePrompt = null) {
            RenderTexture.name = prompt;
            var image = await _asyncPipeline.GenerateAsync(
                prompt,
                width: width,
                height: height,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                negativePrompt: negativePrompt
            );
            TextureConverter.RenderToTexture(image, RenderTexture);
            return _sdPipeline.GetParameters();
        }

        public void Dispose() {
            _sdPipeline?.Dispose();
            _asyncPipeline?.Dispose();
            if (RenderTexture != null) {
                RenderTexture.Release();
            }
        }
    }
}
