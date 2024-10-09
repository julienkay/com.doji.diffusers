using System;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public class CLIPOutput : ModelOutput {

    }
    
    public class TextEncoder : IModel<TextEncoderConfig>, IDisposable {

        public TextEncoderConfig Config { get; }

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        private BackendType Backend { get; set; } = BackendType.GPUCompute;

        /// <summary>
        /// The runtime model.
        /// </summary>
        private Model _model;
        private Worker _worker;
        private ModelOutput _output;

        public TextEncoder(Model model, TextEncoderConfig config, BackendType backend = BackendType.GPUCompute) {
            Config = config ?? new TextEncoderConfig();
            Backend = backend;
            InitializeNetwork(model);
            _output = new ModelOutput();
        }

        private void InitializeNetwork(Model textEncoder) {
            if (textEncoder == null) {
                throw new ArgumentException("TextEncoder Model was null", nameof(textEncoder));
            }

            _model = textEncoder;
            _worker = new Worker(_model, Backend);
        }

        public ModelOutput Execute(Tensor<int> inputIds) {
            if (inputIds is null) {
                throw new ArgumentNullException(nameof(inputIds));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            _worker.Schedule(inputIds);
            _output.GetOutputs(_model, _worker);
            return _output;
        }

        public async Task<ModelOutput> ExecuteAsync(Tensor<int> inputIds) {
            if (inputIds is null) {
                throw new ArgumentNullException(nameof(inputIds));
            }
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            if (_worker == null) {
                throw new NullReferenceException($"{nameof(_worker)} was null");
            }

            var schedule = _worker.ScheduleIterable(inputIds);
            int i = 0;
            while (schedule.MoveNext()) {
                if (++i % 100 == 0) {
                    await Task.Yield();
                }
            }

            _output.GetOutputs(_model, _worker);
            return _output;
        }

        public Tensor CopyOutput(int index) {
            // wrap around to allow for negative indexing
            index = (_output.Count + (index % _output.Count)) % _output.Count;
            Tensor output = null;
            _worker.CopyOutput(index, ref output);
            return output;
        }

        /// <summary>
        /// Instantiate a TextEncoder from a JSON configuration file.
        /// </summary>
        public static TextEncoder FromPretrained(ModelFile modelFile, ModelFile configFile, BackendType backend) {
            return IModel<TextEncoderConfig>.FromPretrained<TextEncoder>(modelFile, configFile, backend);
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}