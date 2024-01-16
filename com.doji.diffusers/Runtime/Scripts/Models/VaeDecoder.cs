using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public class VaeDecoder : IDisposable {

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        private BackendType Backend { get; set; } = BackendType.GPUCompute;

        /// <summary>
        /// The runtime model.
        /// </summary>
        private Model _model;

        private IWorker _worker;
        private Ops _ops;

        public VaeDecoder(ModelAsset modelAsset, BackendType backend = BackendType.GPUCompute) {
            Backend = backend;
            InitializeNetwork(modelAsset);
        }

        private void InitializeNetwork(ModelAsset vaeDecoder) {
            if (vaeDecoder == null) {
                throw new ArgumentException("VaeDecoder ModelAsset was null", nameof(vaeDecoder));
            }

            _model = ModelLoader.Load(vaeDecoder);
            Resources.UnloadAsset(vaeDecoder);

            _worker = WorkerFactory.CreateWorker(Backend, _model);
            _ops = WorkerFactory.CreateOps(Backend, null);
        }

        public TensorFloat ExecuteModel(TensorFloat latentSample) {
            if (latentSample is null) {
                throw new ArgumentNullException(nameof(latentSample));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            _worker.Execute(latentSample);
            TensorFloat sample = _worker.PeekOutput("sample") as TensorFloat;
            TensorFloat image_div_2 = _ops.Div(sample, 2);
            TensorFloat normalized = _ops.Add(image_div_2, 0.5f);
            TensorFloat image = _ops.Clip(normalized, 0.0f, 1.0f);
            return image;
        }

        public void Dispose() {
            _worker?.Dispose();
            _ops?.Dispose();
        }
    }
}