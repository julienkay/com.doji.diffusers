using System;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    internal class Unet : IDisposable {

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        private BackendType Backend { get; set; } = BackendType.GPUCompute;

        /// <summary>
        /// The runtime model.
        /// </summary>
        private Model _model;

        private IWorker _worker;
        private Dictionary<string, Tensor> _inputs = new Dictionary<string, Tensor>();

        public Unet(ModelAsset modelAsset, BackendType backend = BackendType.GPUCompute) {
            Backend = backend;
            InitializeNetwork(modelAsset);
        }

        private void InitializeNetwork(ModelAsset textEncoder) {
            InitializeTextEncoder(textEncoder);
        }

        private void InitializeTextEncoder(ModelAsset unet) {
            if (unet == null) {
                throw new ArgumentException("Unet ModelAsset was null", nameof(unet));
            }

            _model = ModelLoader.Load(unet);
            Resources.UnloadAsset(unet);
            _worker = WorkerFactory.CreateWorker(Backend, _model);
        }

        public TensorFloat ExecuteModel(TensorFloat latentInputTensor, TensorInt timestep, TensorFloat promptEmbeds) {
            if (latentInputTensor is null) {
                throw new ArgumentNullException(nameof(latentInputTensor));
            }
            if (timestep is null) {
                throw new ArgumentNullException(nameof(timestep));
            }
            if (promptEmbeds is null) {
                throw new ArgumentNullException(nameof(promptEmbeds));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            _inputs["sample"] = latentInputTensor;
            _inputs["timestep"] = timestep;
            _inputs["encoder_hidden_states"] = promptEmbeds;

            _worker.Execute(_inputs);
            return _worker.PeekOutput("out_sample") as TensorFloat;
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}