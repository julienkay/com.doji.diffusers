using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public class Unet : IModel<UnetConfig>, IDisposable {

        public UnetConfig Config { get; }

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        private BackendType Backend { get; set; } = BackendType.GPUCompute;

        /// <summary>
        /// The runtime model.
        /// </summary>
        private Model _model;

        private Worker _worker;
        private Dictionary<string, Tensor> _inputs = new Dictionary<string, Tensor>();

        public Unet(Model model, UnetConfig config, BackendType backend = BackendType.GPUCompute) {
            Config = config ?? new UnetConfig();
            Backend = backend;
            InitializeNetwork(model);
        }

        private void InitializeNetwork(Model unet) {
            if (unet == null) {
                throw new ArgumentException("Unet Model was null", nameof(unet));
            }

            _model = unet;
            _worker = new Worker(_model, Backend);
        }

        public Tensor<float> Execute(
            Tensor<float> sample,
            Tensor timestep,
            Tensor<float> encoderHiddenStates,
            Tensor textEmbeds = null,
            Tensor timeIds = null)
        {
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
            if (timestep.dataType != _model.inputs[1].dataType) {
                throw new ArgumentException($"This unet models expects timesteps with data type '{_model.inputs[1].dataType}'. " +
                    $"The timesteps your scheduler provided were of type '{timestep.dataType}'. " +
                    $"Make sure to use a scheduler that is supported for this model.");
            }

            _inputs["sample"] = sample;
            _inputs["timestep"] = timestep;
            _inputs["encoder_hidden_states"] = encoderHiddenStates;
            if (textEmbeds != null) {
                _inputs["text_embeds"] = textEmbeds;
            }
            if (timeIds != null) {
                _inputs["time_ids"] = timeIds;
            }

            _worker.Schedule(_inputs);
            return _worker.PeekOutput("out_sample") as Tensor<float>;
        }

        public async Task<Tensor<float>> ExecuteAsync(
            Tensor<float> sample,
            Tensor timestep,
            Tensor<float> encoderHiddenStates,
            Tensor textEmbeds = null,
            Tensor timeIds = null)
        {
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
            if (timestep.dataType != _model.inputs[1].dataType) {
                throw new ArgumentException($"This unet models expects timesteps with data type '{_model.inputs[1].dataType}'. " +
                    $"The timesteps your scheduler provided were of type '{timestep.dataType}'. " +
                    $"Make sure to use a scheduler that is supported for this model.");
            }

            _inputs["sample"] = sample;
            _inputs["timestep"] = timestep;
            _inputs["encoder_hidden_states"] = encoderHiddenStates;
            if (textEmbeds != null) {
                _inputs["text_embeds"] = textEmbeds;
            }
            if (timeIds != null) {
                _inputs["time_ids"] = timeIds;
            }

            var schedule = _worker.ScheduleIterable(_inputs);
            int i = 0;
            while (schedule.MoveNext()) {
                if (++i % 300 == 0) {
                    await Task.Yield();
                }
            }

            return _worker.PeekOutput("out_sample") as Tensor<float>;
        }

        public Tensor CreateTimestep(TensorShape shape, float t) {
            if (_model == null) {
                throw new NullReferenceException($"{nameof(_model)} was null");
            }
            DataType dType = _model.inputs[1].dataType;
            switch (dType) {
                case DataType.Float:
                    return new Tensor<float>(shape, ArrayUtils.Full(shape.length, (float)t));
                case DataType.Int:
                    return new Tensor<int>(shape, ArrayUtils.Full(shape.length, (int)t));
                default:
                    throw new ArgumentException($"This unet models expects timesteps with data type '{_model.inputs[1].dataType}', " +
                        $"which is not supported yet.");
            }
        }

        /// <summary>
        /// Instantiate a Unet from a JSON configuration file.
        /// </summary>
        public static Unet FromPretrained(DiffusionModel model, BackendType backend) {
            return IModel<UnetConfig>.FromPretrained<Unet>(model.Unet, model.UnetConfig, backend);
        }

        public void Dispose() {
            _worker?.Dispose();
        }
    }
}