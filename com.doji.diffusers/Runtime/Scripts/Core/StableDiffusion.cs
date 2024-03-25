using System;
using System.Threading.Tasks;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// This class wraps various <see cref="DiffusionPipeline"/>s.
    /// It provides a higher level API than the pipeline classes, making it more convenient
    /// to use especially for getting started with doing simple generations.
    /// - you don't have to deal with Tensors, the API uses RenderTexture/Texture2D objects
    /// to return results and sometimes to take image inputs
    /// - You can use txt2img, img2img, etc. and don't need to handle individual pipelines
    /// For more control you can still use a pipeline directly
    /// (use <see cref="DiffusionPipeline.FromPretrained(DiffusionModel, BackendType)"/>)
    /// </summary>
    public class StableDiffusion : IDisposable {

        public DiffusionModel Model {
            get => _model;
            set {
                if (_model != value) {
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
                    _backend = value;
                    Initialize();
                }
            }
        }
        private BackendType _backend = BackendType.GPUCompute;

        private DiffusionPipeline _txt2img;
        private IImg2ImgPipeline _img2img;
        public RenderTexture Result;

        private const int WIDTH = 512;
        private const int HEIGHT = 512;

        public StableDiffusion(DiffusionModel model) {
            _model = model;
            Initialize();
        }

        public StableDiffusion(string modelId) {
            _model = new DiffusionModel(modelId);
            Initialize();
        }

        /// <summary>
        /// Initialise a Stable diffusion pipeline for the current <see cref="Model"/>
        /// and the current <see cref="Backend"/>.
        /// </summary>
        private void Initialize() {
            _txt2img?.Dispose();
            _img2img?.Dispose();
            _txt2img = DiffusionPipeline.FromPretrained(Model, Backend);
            _img2img = _txt2img.As<IImg2ImgPipeline>();
            if (Result == null) {
                Result = new RenderTexture(WIDTH, HEIGHT, 0, RenderTextureFormat.ARGB32);
            }
        }

        /// <summary>
        /// txt2img generation
        /// </summary>
        public Metadata Imagine(string prompt, int width, int height, int numInferenceSteps = 50, float guidanceScale = 7.5f, string negativePrompt = null) {
            Result.name = prompt;

            Profiler.BeginSample("StableDiffusion.Imagine");
            var image = _txt2img.Generate(
                prompt,
                width: width,
                height: height,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                negativePrompt: negativePrompt
            );
            Profiler.EndSample();

            Profiler.BeginSample("Convert to RenderTexture");
            TextureConverter.RenderToTexture(image, Result);
            Profiler.EndSample();

            return _txt2img.GetMetadata();
        }

        public async Task<Metadata> ImagineAsync(string prompt, int width, int height, int numInferenceSteps = 50, float guidanceScale = 7.5f, string negativePrompt = null) {
            Result.name = prompt;
            var image = await _txt2img.GenerateAsync(
                prompt,
                width: width,
                height: height,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                negativePrompt: negativePrompt
            );
            TextureConverter.RenderToTexture(image, Result);
            return _txt2img.GetMetadata();
        }

        /// <summary>
        /// img2img generation
        /// </summary>
        public Metadata Imagine(string prompt, Texture2D inputTexture, int numInferenceSteps = 50, float guidanceScale = 7.5f, string negativePrompt = null, float strength = 0.8f) {
            Result.name = prompt;
            using var input = TextureConverter.ToTensor(inputTexture);

            Profiler.BeginSample("StableDiffusion.Imagine");
            var image = _img2img.Generate(
                prompt: prompt,
                image: input,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                negativePrompt: negativePrompt,
                strength: strength
            );
            Profiler.EndSample();

            Profiler.BeginSample("Convert to RenderTexture");
            TextureConverter.RenderToTexture(image, Result);
            Profiler.EndSample();

            return _img2img.GetMetadata();
        }

        public void Dispose() {
            _txt2img?.Dispose();
            _img2img?.Dispose();
            if (Result != null) {
                Result.Release();
            }
        }
    }
}
