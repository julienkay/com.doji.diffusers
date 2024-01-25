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

        public TextEncoder(Model model, BackendType backend = BackendType.GPUCompute) {
            Backend = backend;
            InitializeNetwork(model);
        }

        private void InitializeNetwork(Model textEncoder) {
            if (textEncoder == null) {
                throw new ArgumentException("TextEncoder Model was null", nameof(textEncoder));
            }

            _model = textEncoder;
            _worker = WorkerFactory.CreateWorker(Backend, _model);
            _allocator = new TensorCachingAllocator();
        }

        public TensorFloat ExecuteModel(TensorInt inputIds) {
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
            return _worker.PeekOutput("last_hidden_state") as TensorFloat;
        }

        public void Dispose() {
            _worker?.Dispose();
            _allocator?.Dispose();
            _ops?.Dispose();
        }
    }
}