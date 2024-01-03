using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using Unity.Sentis;
using Unity.Sentis.Layers;
using UnityEngine;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion pipeline 
    /// </summary>
    /// <remarks>
    /// stable_diffusion/pipeline_onnx_stable_diffusion.py from huggingface/diffusers
    /// </remarks>
    public class StableDiffusionPipeline : IDisposable {

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        private BackendType Backend { get; set; } = BackendType.GPUCompute;

        /// <summary>
        /// The runtime model.
        /// </summary>
        private Model _model;

        private IWorker _worker;
        private ITensorAllocator _allocator;
        private Ops _ops;

        private ClipTokenizer _tokenizer;
 
        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            ModelAsset textEncoder,
            ClipTokenizer tokenizer,
            BackendType backend = BackendType.GPUCompute)
        {
            Backend = backend;
            _tokenizer = tokenizer;
            InitializeNetwork(textEncoder);
        }

        private void InitializeNetwork(ModelAsset textEncoder) {

            InitializeTextEncoder(textEncoder);
        }

        private void InitializeTextEncoder(ModelAsset textEncoder) {
            if (textEncoder == null) {
                throw new ArgumentException("TextEncoder ModelAsset was null", nameof(textEncoder));
            }

            _model = ModelLoader.Load(textEncoder);
            Resources.UnloadAsset(textEncoder);
            _worker = WorkerFactory.CreateWorker(Backend, _model);
            _allocator = new TensorCachingAllocator();
        }

        private void Execute(string prompt) {
            var promptEmbeds = EncodePrompt(prompt);
        }

        private int EncodePrompt(string prompt) {
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            }

            var text_inputs = _tokenizer.EncodePrompt(
                prompt,
                padding: Padding.MaxLength,
                maxLength: _tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            );

            throw new NotImplementedException();
        }

        private int EncodePrompt(List<string> prompt) {
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            }

            throw new NotImplementedException();
        }

        public void Dispose() {
            _worker?.Dispose();
            _allocator?.Dispose();
            _ops?.Dispose();
        }
    }
}
