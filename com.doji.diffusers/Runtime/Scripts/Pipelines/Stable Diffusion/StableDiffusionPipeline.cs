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
        private int _numImagesPerPrompt;

        private Ops _ops;

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
            _vaeDecoder = new VaeDecoder(vaeDecoder, backend);
            _tokenizer = tokenizer;
            _textEncoder = new TextEncoder(textEncoder, backend);
            _scheduler = scheduler;
            _unet = new Unet(unet, backend);
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        /// <inheritdoc cref="Generate(object, int, int, int, float, int, Action{int, int, float[]})"/>
        public TensorFloat Generate(
            string prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            int numImagesPerPrompt = 1,
            Action<int, int, float[]> callback = null)
        {
            return Generate((TextInput)prompt, height, width, numInferenceSteps, guidanceScale, numImagesPerPrompt, callback);
        }

        /// <param name="prompt">The prompts used to generate the batch of images for.</param>
        /// <inheritdoc cref="Generate(object, int, int, int, float, int, Action{int, int, float[]})"/>
        public TensorFloat Generate(
            List<string> prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            int numImagesPerPrompt = 1,
            Action<int, int, float[]> callback = null)
        {
            return Generate((BatchInput)prompt, height, width, numInferenceSteps, guidanceScale, numImagesPerPrompt, callback);
        }

        /// <summary>
        /// Execute the pipeline to generate images.
        /// </summary>
        /// <param name="prompt">The prompt or prompts to guide the image generation.
        /// If not defined, one has to pass `prompt_embeds` instead.</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="numInferenceSteps"> The number of denoising steps.
        /// More denoising steps usually lead to a higher quality image
        /// at the expense of slower inference.</param>
        /// <param name="guidanceScale">Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
        /// `guidance_scale` is defined as `w` of equation 2. of[Imagen Paper] (https://arxiv.org/pdf/2205.11487.pdf).
        /// Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
        /// that are closely linked to the text `prompt`, usually at the expense of lower image quality.</param>
        /// <param name="numImagesPerPrompt">The number of images to generate per prompt.</param>
        /// <param name="callback">A function that will be called at every step during inference.
        /// The function will be called with the following arguments:
        /// `callback(step: int, timestep: int, latents: torch.FloatTensor)`.</param>
        public TensorFloat Generate(
            Input prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            int numImagesPerPrompt = 1,
            Action<int, int, float[]> callback = null)
        {
            _height = height;
            _width = width;
            _numImagesPerPrompt = numImagesPerPrompt;
            CheckInputs();

            if (prompt != null && prompt is TextInput) {
                _batchSize = 1;
            } else if (prompt != null && prompt is BatchInput prompts) {
                _batchSize = prompts.Sequence.Count;
            } else if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            } else {
                throw new ArgumentException($"Invalid prompt argument {nameof(prompt)}");
            }

            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            TensorFloat promptEmbeds = EncodePrompt(prompt, numImagesPerPrompt, doClassifierFreeGuidance);
            _scheduler.SetTimesteps(numInferenceSteps);

            // get the initial random noise
            float[] latents = GenerateLatents();

            for (int i = 0; i < _scheduler.Timesteps.Length; i++) {
                int t = _scheduler.Timesteps[i];

                // expand the latents if doing classifier free guidance
                float[] latentModelInput = doClassifierFreeGuidance ? latents.Tile(2) : latents;
                latentModelInput = _scheduler.ScaleModelInput(latentModelInput, t);
                using TensorFloat latentInputTensor = new TensorFloat(GetLatentsShape(), latentModelInput);

                // predict the noise residual
                using TensorInt timestep = new TensorInt(new TensorShape(_batchSize), ArrayUtils.Full(_batchSize, t));
                TensorFloat noisePred = _unet.ExecuteModel(latentInputTensor, timestep, promptEmbeds);

                // perform guidance
                if (doClassifierFreeGuidance) {
                    TensorFloat noisePredUncond = _ops.Split(noisePred, 0, 0, 2);
                }

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
                using TensorFloat latentSample = new TensorFloat(GetLatentsShape(), latents);
                TensorFloat image = _vaeDecoder.ExecuteModel(latentSample);
                return image;
            }
        }

        private void CheckInputs() {
            if (_height % 8 != 0 || _width % 8 != 0) {
                throw new ArgumentException($"`height` and `width` have to be divisible by 8 but are {_height} and {_width}.");
            }
            if (_numImagesPerPrompt > 1) {
                throw new ArgumentException($"More than one image per prompt not supported yet. `numImagesPerPrompt` was {_numImagesPerPrompt}.");
            }
        }

        private TensorFloat EncodePrompt(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            object negativePrompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null)
        {

            if (promptEmbeds == null) {
                var textInputs = _tokenizer.Encode(
                    text: prompt,
                    padding: Padding.MaxLength,
                    maxLength: _tokenizer.ModelMaxLength,
                    truncation: Truncation.LongestFirst
                ) as InputEncoding;
                int[] textInputIds = textInputs.InputIds.ToArray() ?? throw new Exception("Failed to get input ids from tokenizer.");

                using TensorInt textIdTensor = new TensorInt(new TensorShape(_batchSize, textInputIds.Length), textInputIds);
                promptEmbeds = _textEncoder.ExecuteModel(textIdTensor);
            }

            // get unconditional embeddings for classifier free guidance
            if (doClassifierFreeGuidance && negativePromptEmbeds == null) {
                List<string> uncondTokens;
                if (negativePrompt == null) {
                    uncondTokens = Enumerable.Repeat("", _batchSize).ToList();
                } else if (prompt.GetType() != negativePrompt.GetType()) {
                    throw new ArgumentException($"`negativePrompt` should be the same type as `prompt`, but got {negativePrompt.GetType()} != {prompt.GetType()}.");
                } else if (negativePrompt is string) {
                    uncondTokens = Enumerable.Repeat(negativePrompt as string, _batchSize).ToList();
                } else if (_batchSize != negativePromptEmbeds.shape.length) {
                    throw new ArgumentException($"`negativePrompt`: {negativePrompt} has batch size {negativePromptEmbeds.shape.length}, " +
                        $"but `prompt`: {promptEmbeds} has batch size {_batchSize}. Please make sure that passed `negativePrompt` matches " +
                        $"the batch size of `prompt`.");
                } else {
                    uncondTokens = negativePrompt as List<string>;
                }

                int maxLength = promptEmbeds.shape[1];
                var uncondInput = _tokenizer.Encode<BatchInput>(
                    text: uncondTokens,
                    padding: Padding.MaxLength,
                    maxLength: maxLength,
                    truncation: Truncation.LongestFirst
                ) as BatchEncoding;
                int[] uncondInputIds = uncondInput.InputIds as int[] ?? throw new Exception("Failed to get unconditioned input ids.");

                using TensorInt uncondIdTensor = new TensorInt(new TensorShape(_batchSize, uncondInputIds.Length), uncondInputIds);
                negativePromptEmbeds = _textEncoder.ExecuteModel(uncondIdTensor);
            }

            if (doClassifierFreeGuidance) {
                // For classifier free guidance, we need to do two forward passes.
                // Here we concatenate the unconditional and text embeddings into a single batch
                // to avoid doing two forward passes
                promptEmbeds = _ops.Concat(new Tensor[] { negativePromptEmbeds, promptEmbeds }, 0) as TensorFloat;
            }

            return promptEmbeds;
        }

        private float[] GenerateLatents() {
            int size = GetLatentsShape().length;
            return ArrayUtils.Randn(size, 0, _scheduler.InitNoiseSigma);
        }

        private TensorShape GetLatentsShape() {
            return new TensorShape(
                _batchSize * _numImagesPerPrompt,
                4, // unet.in_channels
                _height / 8,
                _width / 8);
        }

        public void Dispose() {
            _textEncoder?.Dispose();
            _vaeDecoder?.Dispose();
            _textEncoder?.Dispose();
            _unet?.Dispose();
            _ops?.Dispose();
        }
    }
}
