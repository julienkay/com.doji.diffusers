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

        public VaeEncoder(Model model, VaeConfig config, BackendType backend = BackendType.GPUCompute) {
            Config = config ?? new VaeConfig();
            Backend = backend;
            InitializeNetwork(model);
        }

        private void InitializeNetwork(Model vaeEncoder) {
            if (vaeEncoder == null) {
                throw new ArgumentException("VaeEncoder Model was null", nameof(vaeEncoder));
            }

            _model = vaeEncoder;
            _worker = WorkerFactory.CreateWorker(Backend, _model);
        }

        /// <summary>
        /// Encodes the image.
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
            return _worker.PeekOutput("latent_sample") as TensorFloat;
        }

        /// <summary>
        /// Instantiate a VaeEncoder from a JSON configuration file.
        /// </summary>
        public static VaeEncoder FromPretrained(DiffusionModel model, BackendType backend) {
            return IModel<VaeConfig>.FromPretrained<VaeEncoder>(model.VaeEncoder, model.VaeEncoderConfig, backend);
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}