using System;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public class VaeDecoder : IModel<VaeConfig>, IDisposable {

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

        public VaeDecoder(Model model, VaeConfig config, BackendType backend = BackendType.GPUCompute) {
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

        public TensorFloat Execute(TensorFloat latentSample) {
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
            TensorFloat normalized = _ops.Mad(sample, 0.5f, 0.5f);
            TensorFloat image = _ops.Clip(normalized, 0.0f, 1.0f);
            return image;
        }

        public async Task<TensorFloat> ExecuteAsync(TensorFloat latentSample) {
            if (latentSample is null) {
                throw new ArgumentNullException(nameof(latentSample));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            var schedule = _worker.StartManualSchedule(latentSample);
            int i = 0;
            while (schedule.MoveNext()) {
                if (++i % 50 == 0) {
                    await Task.Yield();
                }
            }

            TensorFloat sample = _worker.PeekOutput("sample") as TensorFloat;
            TensorFloat normalized = _ops.Mad(sample, 0.5f, 0.5f);
            TensorFloat image = _ops.Clip(normalized, 0.0f, 1.0f);
            return image;
        }

        /// <summary>
        /// Instantiate a VaeDecoder from a JSON configuration file.
        /// </summary>
        public static VaeDecoder FromPretrained(ModelFile vaeConfig, BackendType backend) {
            return IModel<VaeConfig>.FromPretrained<VaeDecoder>(vaeConfig, backend);
        }

        public void Dispose() {
            _worker?.Dispose();
            _ops?.Dispose();
        }
    }
}