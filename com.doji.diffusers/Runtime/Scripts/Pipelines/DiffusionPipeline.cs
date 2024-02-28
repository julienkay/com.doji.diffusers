using Doji.AI.Transformers;
using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public abstract partial class DiffusionPipeline {

        public string NameOrPath { get; protected set; }
        public PipelineConfig Config { get; protected set; }

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

        protected void CheckInputs(uint? seed) {
            if (_height % 8 != 0 || _width % 8 != 0) {
                throw new ArgumentException($"`height` and `width` have to be divisible by 8 but are {_height} and {_width}.");
            }
            if (_numImagesPerPrompt > 1) {
                throw new ArgumentException($"More than one image per prompt not supported yet. `numImagesPerPrompt` was {_numImagesPerPrompt}.");
            }
            if (_latents != null && seed != null) {
                throw new ArgumentException($"Both a seed and pre-generated noise has been passed. Please use either one or the other.");
            }
        }
    }
}