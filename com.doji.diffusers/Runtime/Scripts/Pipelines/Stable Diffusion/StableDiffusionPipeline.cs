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

        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            ModelAsset textEncoder,
            ClipTokenizer tokenizer,
            BackendType backend = BackendType.GPUCompute)
        {
            _tokenizer = tokenizer;
            _textEncoder = new TextEncoder(textEncoder);
        }

        private void Execute(string prompt) {
            var promptEmbeds = EncodePrompt(prompt);
        }

        private int EncodePrompt(string prompt) {
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            }

            var text_inputs = _tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: _tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            );
            List<int> textInputIds = text_inputs.InputIds ?? throw new Exception("Failed to get input ids from tokenizer.");

            TensorInt tensor = new TensorInt(new TensorShape(1, textInputIds.Count), textInputIds.ToArray());
            Tensor promptEmbeds = _textEncoder.ExecuteModel(tensor);


            throw new NotImplementedException();
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
