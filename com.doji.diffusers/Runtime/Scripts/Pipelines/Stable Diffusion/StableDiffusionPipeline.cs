using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion pipeline 
    /// </summary>
    /// <remarks>
    /// stable_diffusion/pipeline_onnx_stable_diffusion.py from huggingface/diffusers
    /// </remarks>
    public class StableDiffusionPipeline : IDisposable {

        private ClipTokenizer _tokenizer;
        private TextEncoder _textEncoder;
        private PNDMScheduler _scheduler;
        private Unet _unet;

        private int _width;
        private int _height;
        private int _batchSize;

        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            ModelAsset textEncoder,
            ClipTokenizer tokenizer,
            PNDMScheduler scheduler,
            Unet unet,
            BackendType backend = BackendType.GPUCompute)
        {
            _tokenizer = tokenizer;
            _textEncoder = new TextEncoder(textEncoder);
            _scheduler = scheduler;
            _unet = unet;
        }

        public void Execute(
            string prompt,
            int width = 512,
            int height = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f)
        {
            _width = width;
            _height = height;
            _batchSize = 1;
            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            TensorFloat promptEmbeds = EncodePrompt(prompt, doClassifierFreeGuidance);
            _scheduler.SetTimesteps(numInferenceSteps);
            float[] latents = GenerateLatents();

            for (int i = 0; i < _scheduler.Timesteps.Length; i++) {
                int t = _scheduler.Timesteps[i];

                // expand the latents if doing classifier free guidance
                float[] latentModelInput = doClassifierFreeGuidance ? latents.Repeat() : latents;
                latentModelInput = _scheduler.ScaleModelInput(latentModelInput, t);
                TensorFloat latentInputTensor = new TensorFloat(GetLatentsShape(), latentModelInput);

                // predict the noise residual
                TensorInt timestep = _batchSize == 1 ?
                    new TensorInt(t) :
                    new TensorInt(new TensorShape(_batchSize), ArrayUtils.Full(_batchSize, t));
                Tensor noisePred = _unet.ExecuteModel(latentInputTensor, timestep, promptEmbeds);
            }

        }

        private TensorFloat EncodePrompt(
            string prompt,
            bool doClassifierFreeGuidance)
        {
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            }

            int batchSize = 1;

            var text_inputs = _tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: _tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            );
            List<int> textInputIds = text_inputs.InputIds ?? throw new Exception("Failed to get input ids from tokenizer.");

            TensorInt tensor = new TensorInt(new TensorShape(batchSize, textInputIds.Count), textInputIds.ToArray());
            TensorFloat promptEmbeds = _textEncoder.ExecuteModel(tensor) as TensorFloat;
            tensor.Dispose();
            return promptEmbeds;
        }

        private int EncodePrompt(List<string> prompt) {
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            }

            throw new NotImplementedException();
        }

        private float[] GenerateLatents() {
            int size = _batchSize * 4 * _width * _height;
            return ArrayUtils.Randn(size, 0, _scheduler.InitNoiseSigma);
        }

        private TensorShape GetLatentsShape() {
            return new TensorShape()
        }

        public void Dispose() {
            _textEncoder?.Dispose();
        }
    }
}
