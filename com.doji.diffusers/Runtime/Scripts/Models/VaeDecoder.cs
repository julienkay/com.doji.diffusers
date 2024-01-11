using System;
using System.Collections.Generic;
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

            // bake normallization into model
            // would rather use temp ops, but that leaks memory
            string output = _model.outputs[0];
            _model.layers.Add(new Unity.Sentis.Layers.ConstantOfShape("TWO", output, 2f));
            _model.layers.Add(new Unity.Sentis.Layers.ConstantOfShape("HALF", output, 0.5f));
            _model.layers.Add(new Unity.Sentis.Layers.Div("out / 2", output, "TWO"));
            _model.layers.Add(new Unity.Sentis.Layers.Add("sample_normalized", "out / 2", "HALF"));
            _model.outputs = new List<string>() { output, "sample_normalized" };

            _worker = WorkerFactory.CreateWorker(Backend, _model);
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
            return _worker.PeekOutput("sample_normalized") as TensorFloat;
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}