using System;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public class Unet : IDisposable {

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

        public Unet(Model model, BackendType backend = BackendType.GPUCompute) {
            Backend = backend;
            InitializeNetwork(model);
        }

        private void InitializeNetwork(Model unet) {
            if (unet == null) {
                throw new ArgumentException("Unet Model was null", nameof(unet));
            }

            _model = unet;
            _worker = WorkerFactory.CreateWorker(Backend, _model);
        }

        public TensorFloat ExecuteModel(TensorFloat sample, TensorInt timestep, TensorFloat encoderHiddenStates) {
            if (sample is null) {
                throw new ArgumentNullException(nameof(sample));
            }
            if (timestep is null) {
                throw new ArgumentNullException(nameof(timestep));
            }
            if (encoderHiddenStates is null) {
                throw new ArgumentNullException(nameof(encoderHiddenStates));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            _inputs["sample"] = sample;
            _inputs["timestep"] = timestep;
            _inputs["encoder_hidden_states"] = encoderHiddenStates;

            _worker.Execute(_inputs);
            return _worker.PeekOutput("out_sample") as TensorFloat;
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}