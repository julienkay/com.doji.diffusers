using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion pipeline 
    /// </summary>
    /// <remarks>
    /// stable_diffusion/pipeline_onnx_stable_diffusion.py from huggingface/diffusers
    /// </remarks>
    public class StableDiffusionPipeline : IDisposable {

        private VaeDecoder _vaeDecoder;
        private ClipTokenizer _tokenizer;
        private TextEncoder _textEncoder;
        private PNDMScheduler _scheduler;
        private Unet _unet;

        private int _height;
        private int _width;
        private int _batchSize;

        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            ModelAsset vaeDecoder,
            ModelAsset textEncoder,
            ClipTokenizer tokenizer,
            PNDMScheduler scheduler,
            ModelAsset unet,
            BackendType backend = BackendType.GPUCompute)
        {
            _vaeDecoder = new VaeDecoder(vaeDecoder);
            _tokenizer = tokenizer;
            _textEncoder = new TextEncoder(textEncoder);
            _scheduler = scheduler;
            _unet = new Unet(unet);
        }

        public TensorFloat Execute(
            string prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            Action<int, int, float[]> callback = null)
        {
            _height = height;
            _width = width;
            _batchSize = 1;
            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            TensorFloat promptEmbeds = EncodePrompt(prompt, doClassifierFreeGuidance);
            _scheduler.SetTimesteps(numInferenceSteps);

            // get the initial random noise
            float[] latents = GenerateLatents();

            for (int i = 0; i < _scheduler.Timesteps.Length; i++) {
                int t = _scheduler.Timesteps[i];

                // expand the latents if doing classifier free guidance
                float[] latentModelInput = doClassifierFreeGuidance ? latents.Repeat() : latents;
                latentModelInput = _scheduler.ScaleModelInput(latentModelInput, t);
                TensorFloat latentInputTensor = new TensorFloat(GetLatentsShape(), latentModelInput);

                // predict the noise residual
                TensorInt timestep = new TensorInt(new TensorShape(_batchSize), ArrayUtils.Full(_batchSize, t));
                TensorFloat noisePred = _unet.ExecuteModel(latentInputTensor, timestep, promptEmbeds);
                noisePred.MakeReadable();
                float[] noise = noisePred.ToReadOnlyArray();

                // perform guidance
                if (doClassifierFreeGuidance) {
                    float[] noisePredUncond = noise.Take(noise.Length / 2).ToArray();
                    float[] noisePredText = noise.Skip(noise.Length / 2).ToArray();
                    noise = noisePredUncond.Zip(noisePredText, (a, b) => a + guidanceScale * (b - a)).ToArray();
                }

                // compute the previous noisy sample x_t -> x_t-1
                var schedulerOutput = _scheduler.Step(noise, t, latents);
                latents = schedulerOutput.PrevSample;

                callback?.Invoke(i / _scheduler.Order, t, latents);
            }

            for (int l = 0; l < latents.Length; l++) {
                latents[l] = 1.0f / 0.18215f * latents[l];
            }

            // batch
            if (_batchSize > 1) {
                throw new NotImplementedException();
            } else {
                return _vaeDecoder.ExecuteModel(new TensorFloat(GetLatentsShape(), latents));
            }
        }

        private TensorFloat EncodePrompt(
            string prompt,
            bool doClassifierFreeGuidance,
            string negative_prompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null)
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
            promptEmbeds = _textEncoder.ExecuteModel(tensor) as TensorFloat;
            tensor.Dispose();

            // get unconditional embeddings for classifier free guidance
            if (doClassifierFreeGuidance && negativePromptEmbeds == null) {
                UnityEngine.Debug.LogError("TODO: implement classifier free guidance not implemented yet. Ignoring...");
                return promptEmbeds;
            }

            if (doClassifierFreeGuidance) {
                UnityEngine.Debug.LogError("TODO: implement classifier free guidance not implemented yet. Ignoring...");
                return promptEmbeds;
            }

            return promptEmbeds;
        }

        private int EncodePrompt(List<string> prompt) {
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            }

            throw new NotImplementedException();
        }

        private float[] GenerateLatents() {
            int size = GetLatentsShape().length;
            return ArrayUtils.Randn(size, 0, _scheduler.InitNoiseSigma);
        }

        private TensorShape GetLatentsShape() {
            return new TensorShape(_batchSize, 4, _height, _width);
        }

        public void Dispose() {
            _textEncoder?.Dispose();
        }
    }
}
