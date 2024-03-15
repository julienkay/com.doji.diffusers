using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public class VaeEncoder : IModel<VaeConfig>, IDisposable {

        public VaeConfig Config { get; }

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

        public VaeEncoder(Model model, VaeConfig config, BackendType backend = BackendType.GPUCompute) {
            Config = config ?? new VaeConfig();
            Backend = backend;
            InitializeNetwork(model);
        }

        private void InitializeNetwork(Model vaeDecoder) {
            if (vaeDecoder == null) {
                throw new ArgumentException("VaeDecoder Model was null", nameof(vaeDecoder));
            }

            _model = vaeDecoder;
            _worker = WorkerFactory.CreateWorker(Backend, _model);
            _ops = WorkerFactory.CreateOps(Backend, null);
        }

        /// <summary>
        /// Encodes the image.
        /// TODO: Who's responsible for normalizing input?
        /// </summary>
        public TensorFloat Execute(TensorFloat sample) {
            if (sample is null) {
                throw new ArgumentNullException(nameof(sample));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            _worker.Execute(sample);
            TensorFloat latentSample = _worker.PeekOutput("latent_sample") as TensorFloat;
            TensorFloat normalized = _ops.Mad(latentSample, 0.5f, 0.5f);
            TensorFloat image = _ops.Clip(normalized, 0.0f, 1.0f);
            return latentSample;
        }

        /// <summary>
        /// Instantiate a VaeEncoder from a JSON configuration file.
        /// </summary>
        public static VaeEncoder FromPretrained(ModelFile vaeConfig, BackendType backend) {
            return IModel<VaeConfig>.FromPretrained<VaeEncoder>(vaeConfig, backend);
        }

        public void Dispose() {
            _worker?.Dispose();
            _ops?.Dispose();
        }
    }
}