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

        private Worker _worker;

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
            _worker = new Worker(_model, Backend);
        }

        public Tensor<float> Execute(Tensor<float> latentSample) {
            if (latentSample is null) {
                throw new ArgumentNullException(nameof(latentSample));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            _worker.Schedule(latentSample);
            return _worker.PeekOutput("sample") as Tensor<float>;
        }

        public async Task<Tensor<float>> ExecuteAsync(Tensor<float> latentSample) {
            if (latentSample is null) {
                throw new ArgumentNullException(nameof(latentSample));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            var schedule = _worker.ScheduleIterable(latentSample);
            int i = 0;
            while (schedule.MoveNext()) {
                if (++i % 50 == 0) {
                    await Task.Yield();
                }
            }

            return _worker.PeekOutput("sample") as Tensor<float>;
        }

        /// <summary>
        /// Instantiate a VaeDecoder from a JSON configuration file.
        /// </summary>
        public static VaeDecoder FromPretrained(DiffusionModel model, BackendType backend) {
            return IModel<VaeConfig>.FromPretrained<VaeDecoder>(model.VaeDecoder, model.VaeDecoderConfig, backend);
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}