using Doji.AI.Transformers;
using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public abstract class DiffusionPipelineBase : IDisposable {

        public string NameOrPath { get; protected set; }
        public PipelineConfig Config { get; protected set; }

        public VaeDecoder VaeDecoder { get; protected set; }
        public ClipTokenizer Tokenizer { get; protected set; }
        public TextEncoder TextEncoder { get; protected set; }
        public Scheduler Scheduler { get; protected set; }
        public Unet Unet { get; protected set; }

        protected Input _prompt;
        protected Input _negativePrompt;
        protected int _steps;
        protected int _height;
        protected int _width;
        protected int _batchSize;
        protected int _numImagesPerPrompt;
        protected float _guidanceScale;
        protected float? _eta;
        protected uint? _seed;
        protected TensorFloat _latents;

        protected void CheckInputs() {
            if (_height % 8 != 0 || _width % 8 != 0) {
                throw new ArgumentException($"`height` and `width` have to be divisible by 8 but are {_height} and {_width}.");
            }
            if (_numImagesPerPrompt > 1) {
                throw new ArgumentException($"More than one image per prompt not supported yet. `numImagesPerPrompt` was {_numImagesPerPrompt}.");
            }
            if (_latents != null && _seed != null) {
                throw new ArgumentException($"Both a seed and pre-generated noise has been passed. Please use either one or the other.");
            }
        }

        public Parameters GetParameters() {
            if (_prompt is not SingleInput) {
                throw new NotImplementedException("GetParameters not yet implemented for batch inputs.");
            }

            return new Parameters() {
                PackageVersion = System.Diagnostics.FileVersionInfo.GetVersionInfo(System.Reflection.Assembly.GetExecutingAssembly().Location).ProductVersion,
                Prompt = (_prompt as SingleInput).Text,
                Model = NameOrPath,
                NegativePrompt = _negativePrompt != null ? (_negativePrompt as SingleInput).Text : null,
                Steps = _steps,
                Sampler = Scheduler.GetType().Name,
                CfgScale = _guidanceScale,
                Seed = _seed,
                Width = _width,
                Height = _height,
                Eta = _eta
            };
        }

        public virtual void Dispose() {
            VaeDecoder?.Dispose();
            TextEncoder?.Dispose();
            Scheduler?.Dispose();
            Unet?.Dispose();
        }
    }
}