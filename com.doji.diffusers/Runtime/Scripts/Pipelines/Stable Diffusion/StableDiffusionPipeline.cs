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

        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            ModelAsset textEncoder,
            ClipTokenizer tokenizer,
            PNDMScheduler scheduler,
            BackendType backend = BackendType.GPUCompute)
        {
            _tokenizer = tokenizer;
            _textEncoder = new TextEncoder(textEncoder);
            _scheduler = scheduler;
        }

        public void Execute(
            string prompt,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f)
        {
            int batchSize = 1;
            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            var promptEmbeds = EncodePrompt(prompt, doClassifierFreeGuidance);
            var latents = ArrayUtils.Randn(batchSize * 4 * 512 * 512);

            _scheduler.SetTimesteps(numInferenceSteps);
        }

        private Tensor EncodePrompt(
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
            return promptEmbeds;
        }

        private int EncodePrompt(List<string> prompt) {
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            }

            throw new NotImplementedException();
        }

        public void Dispose() {
            _textEncoder?.Dispose();
        }
    }
}
