using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public class TextEncoder : IDisposable {

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

        public TextEncoder(ModelAsset modelAsset, BackendType backend = BackendType.GPUCompute) {
            Backend = backend;
            InitializeNetwork(modelAsset);
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

        public Tensor ExecuteModel(TensorInt inputIds) {
            if (inputIds is null) {
                throw new ArgumentNullException(nameof(inputIds));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            _worker.Execute(inputIds);
            return _worker.PeekOutput("last_hidden_state");
        }

        public void Dispose() {
            _worker?.Dispose();
            _allocator?.Dispose();
            _ops?.Dispose();
        }
    }
}