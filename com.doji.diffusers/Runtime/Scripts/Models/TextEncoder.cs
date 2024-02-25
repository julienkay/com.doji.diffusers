using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public class CLIPOutput : ModelOutput {

        public new Tensor this[int index] {
            get {
                // wrap around to allow for negative indexing
                index = (Count + (index % Count)) % Count;
                return this[index];
            }
            set {
                this[index] = value;
            }
        }
    }
    
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
        private ModelOutput _output;

        public TextEncoder(Model model, BackendType backend = BackendType.GPUCompute) {
            Backend = backend;
            InitializeNetwork(model);
            _output = new ModelOutput();
        }

        private void InitializeNetwork(Model textEncoder) {
            if (textEncoder == null) {
                throw new ArgumentException("TextEncoder Model was null", nameof(textEncoder));
            }

            _model = textEncoder;
            _worker = WorkerFactory.CreateWorker(Backend, _model);
        }

        public ModelOutput ExecuteModel(TensorInt inputIds) {
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
            _output.GetOutputs(_model, _worker);
            return _output;
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}