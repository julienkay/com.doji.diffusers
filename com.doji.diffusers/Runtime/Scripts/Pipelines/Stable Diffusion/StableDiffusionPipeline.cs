using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion pipeline 
    /// </summary>
    /// <remarks>
    /// stable_diffusion/pipeline_onnx_stable_diffusion.py from huggingface/diffusers
    /// </remarks>
    public partial class StableDiffusionPipeline : DiffusionPipeline, IDisposable {

        public VaeDecoder VaeDecoder { get; private set; }
        public ClipTokenizer Tokenizer { get; private set; }
        public TextEncoder TextEncoder { get; private set; }
        public Scheduler Scheduler { get; private set; }
        public Unet Unet { get; private set; }

        private Input _prompt;
        private Input _negativePrompt;
        private int _steps;
        private int _height;
        private int _width;
        private int _batchSize;
        private int _numImagesPerPrompt;
        private float _guidanceScale;
        private float? _eta;
        private uint? _seed;
        private Tensor _latents;

        private Ops _ops;

        /// <summary>
        /// Initializes a new stable diffusion pipeline.
        /// </summary>
        public StableDiffusionPipeline(
            Model vaeDecoder,
            Model textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Model unet,
            BackendType backend = BackendType.GPUCompute)
        {
            // FIXME: VaeDecoder exceeds the thread group limit with GPU backend,
            // decoding on CPU is much slower, but curiously the outputs with GPUCompute backend
            // seem correct even despite errors?
            VaeDecoder = new VaeDecoder(vaeDecoder, BackendType.GPUCompute);
            Tokenizer = tokenizer;
            TextEncoder = new TextEncoder(textEncoder, backend);
            Scheduler = scheduler;
            Unet = new Unet(unet, backend);
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        /// <inheritdoc cref="Generate(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, int, TensorFloat})"/>
        public TensorFloat Generate(
            string prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            string negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, int, TensorFloat> callback = null)
        {
            return Generate((TextInput)prompt, height, width, numInferenceSteps, guidanceScale, 
               (TextInput)negativePrompt, numImagesPerPrompt, eta, seed, latents, callback);
        }

        /// <param name="prompt">The prompts used to generate the batch of images for.</param>
        /// <inheritdoc cref="Generate(Input, int, int, int, float, Input, int, float, uint?, TensorFloat, Action{int, int, TensorFloat})"/>
        public TensorFloat Generate(
            List<string> prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            List<string> negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, int, TensorFloat> callback = null)
        {
            return Generate((BatchInput)prompt, height, width, numInferenceSteps, guidanceScale,
                (BatchInput)negativePrompt, numImagesPerPrompt, eta, seed, latents, callback);
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
        /// <param name="eta">Corresponds to parameter eta in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
        /// <see cref="DDIMScheduler"/>, will be ignored for others.</param>
        /// <param name="seed">A seed to use to generate initial noise. Set this to make generation deterministic.</param>
        /// <param name="latents">Pre-generated noise, sampled from a Gaussian distribution, to be used as inputs for image
        /// generation. If not provided, a latents tensor will be generated for you using the supplied <paramref name="seed"/>.</param>
        /// <param name="callback">A function that will be called at every step during inference.
        /// The function will be called with the following arguments:
        /// `callback(step: int, timestep: int, latents: torch.FloatTensor)`.</param>
        public TensorFloat Generate(
            Input prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            Input negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, int, TensorFloat> callback = null)
        {
            Profiler.BeginSample($"{GetType().Name}.Generate");

            _prompt = prompt;
            _negativePrompt = negativePrompt;
            _height = height;
            _width = width;
            _numImagesPerPrompt = numImagesPerPrompt;
            _guidanceScale = guidanceScale;
            _eta = eta;
            _seed = seed != null ? seed.Value : unchecked((uint)new System.Random().Next());
            _latents = latents;
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

            Profiler.BeginSample("Encode Prompt(s)");
            TensorFloat promptEmbeds = EncodePrompt(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);
            Profiler.EndSample();

            // get the initial random noise
            TensorShape latentsShape = GetLatentsShape();
            if (latents == null) {
                Profiler.BeginSample("Generate Latents");
                latents = _ops.RandomNormal(latentsShape, 0, 1, _seed);
                Profiler.EndSample();
            } else if (latents.shape != latentsShape) {
                throw new ArgumentException($"Unexpected latents shape, got {latents.shape}, expected {latentsShape}");
            }
            TensorFloat initialLatents = latents;

            Profiler.BeginSample($"{Scheduler.GetType().Name}.SetTimesteps");
            Scheduler.SetTimesteps(numInferenceSteps);
            Profiler.EndSample();

            if (Scheduler.InitNoiseSigma > 1.0f) {
                Profiler.BeginSample("Multiply latents with scheduler sigma");
                latents = _ops.Mul(Scheduler.InitNoiseSigma, latents);
                Profiler.EndSample();
            }

            Profiler.BeginSample($"Denoising Loop");
            for (int i = 0; i < Scheduler.Timesteps.Length; i++) {
                int t = Scheduler.Timesteps[i];

                // expand the latents if doing classifier free guidance
                TensorFloat latentModelInput = doClassifierFreeGuidance ? _ops.Concat(new Tensor[] { latents, latents }, 0) as TensorFloat : latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                Profiler.BeginSample("Prepare Timestep Tensor");
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(_batchSize), t);
                Profiler.EndSample();

                Profiler.BeginSample("Execute Unet");
                TensorFloat noisePred = Unet.ExecuteModel(latentModelInput, timestep, promptEmbeds);
                Profiler.EndSample();

                // perform guidance
                if (doClassifierFreeGuidance) {
                    Profiler.BeginSample("Extend Predicted Noise For Classifier-Free Guidance");

                    int halfLength = noisePred.shape.length / 2;
                    var noisePredUncond = _ops.Split(noisePred, axis: 0, start: 0, end: 1);
                    var noisePredText = _ops.Split(noisePred, axis: 0, start: 1, end: 2);

                    var tmp = _ops.Sub(noisePredText, noisePredUncond);
                    var tmp2 = _ops.Mul(guidanceScale, tmp);
                    noisePred = _ops.Add(noisePredUncond, tmp2);
                    Profiler.EndSample();
                }

                // compute the previous noisy sample x_t -> x_t-1
                Profiler.BeginSample($"{Scheduler.GetType().Name}.Step");
                var schedulerOutput = Scheduler.Step(noisePred, t, latents, eta);
                latents = schedulerOutput.PrevSample;
                Profiler.EndSample();

                callback?.Invoke(i / Scheduler.Order, t, latents);
            }
            Profiler.EndSample();

            Profiler.BeginSample($"Scale Latents");
            TensorFloat result = _ops.Div(latents, 0.18215f);
            Profiler.EndSample();

            // batch decode
            if (_batchSize > 1) {
                throw new NotImplementedException();
            }

            Profiler.BeginSample($"VaeDecoder Decode Image");
            TensorFloat image = VaeDecoder.ExecuteModel(result);
            Profiler.EndSample();

            initialLatents.Dispose();
            Profiler.EndSample();
            return image;
        }

        private void CheckInputs() {
            if (_height % 8 != 0 || _width % 8 != 0) {
                throw new ArgumentException($"`height` and `width` have to be divisible by 8 but are {_height} and {_width}.");
            }
            if (_numImagesPerPrompt > 1) {
                throw new ArgumentException($"More than one image per prompt not supported yet. `numImagesPerPrompt` was {_numImagesPerPrompt}.");
            }
            if (_latents != null && _seed != null) {
                throw new ArgumentException($"Both a seed and pre-generated noise has been passed. Please use either one or the other.");
            }
        }

        private TensorFloat EncodePrompt(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null)
        {

            if (promptEmbeds == null) {
                Profiler.BeginSample("CLIPTokenizer Encode Input");
                var textInputs = Tokenizer.Encode(
                    text: prompt,
                    padding: Padding.MaxLength,
                    maxLength: Tokenizer.ModelMaxLength,
                    truncation: Truncation.LongestFirst
                ) as InputEncoding;
                int[] textInputIds = textInputs.InputIds.ToArray() ?? throw new Exception("Failed to get input ids from tokenizer.");
                Profiler.EndSample();

                Profiler.BeginSample("Prepare Text ID Tensor");
                using TensorInt textIdTensor = new TensorInt(new TensorShape(_batchSize, textInputIds.Length), textInputIds);
                Profiler.EndSample();

                Profiler.BeginSample("Execute TextEncoder");
                promptEmbeds = TextEncoder.ExecuteModel(textIdTensor);
                Profiler.EndSample();
            }
            bool ownsPromptEmbeds = false;
               
            // get unconditional embeddings for classifier free guidance
            if (doClassifierFreeGuidance && negativePromptEmbeds == null) {
                ownsPromptEmbeds = true;
                promptEmbeds.TakeOwnership();

                List<string> uncondTokens;
                if (negativePrompt == null) {
                    uncondTokens = Enumerable.Repeat("", _batchSize).ToList();
                } else if (prompt.GetType() != negativePrompt.GetType()) {
                    throw new ArgumentException($"`negativePrompt` should be the same type as `prompt`, but got {negativePrompt.GetType()} != {prompt.GetType()}.");
                } else if (negativePrompt is SingleInput) {
                    uncondTokens = Enumerable.Repeat((negativePrompt as SingleInput).Text, _batchSize).ToList();
                } else if (_batchSize != negativePromptEmbeds.shape.length) {
                    throw new ArgumentException($"`negativePrompt`: {negativePrompt} has batch size {negativePromptEmbeds.shape.length}, " +
                        $"but `prompt`: {promptEmbeds} has batch size {_batchSize}. Please make sure that passed `negativePrompt` matches " +
                        $"the batch size of `prompt`.");
                } else {
                    uncondTokens = (negativePrompt as BatchInput).Sequence as List<string>;
                }

                Profiler.BeginSample("CLIPTokenizer Encode Unconditioned Input");
                int maxLength = promptEmbeds.shape[1];
                var uncondInput = Tokenizer.Encode<BatchInput>(
                    text: uncondTokens,
                    padding: Padding.MaxLength,
                    maxLength: maxLength,
                    truncation: Truncation.LongestFirst
                ) as BatchEncoding;
                int[] uncondInputIds = uncondInput.InputIds as int[] ?? throw new Exception("Failed to get unconditioned input ids.");
                Profiler.EndSample();

                Profiler.BeginSample("Prepare Unconditioned Text ID Tensor");
                using TensorInt uncondIdTensor = new TensorInt(new TensorShape(_batchSize, uncondInputIds.Length), uncondInputIds);
                Profiler.EndSample();

                Profiler.BeginSample("Execute TextEncoder For Unconditioned Input");
                negativePromptEmbeds = TextEncoder.ExecuteModel(uncondIdTensor);
                Profiler.EndSample();
            }

            if (doClassifierFreeGuidance) {
                // For classifier free guidance, we need to do two forward passes.
                // Here we concatenate the unconditional and text embeddings into a single batch
                // to avoid doing two forward passes
                Profiler.BeginSample("Concat Prompt Embeds For Classifier-Fee Guidance");
                TensorFloat combinedEmbeddings = _ops.Concat(new Tensor[] { negativePromptEmbeds, promptEmbeds }, 0) as TensorFloat;
                Profiler.EndSample();

                if (ownsPromptEmbeds) {
                    promptEmbeds.Dispose();
                }

                return combinedEmbeddings;
            }

            return promptEmbeds;
        }

        private TensorShape GetLatentsShape() {
            return new TensorShape(
                _batchSize * _numImagesPerPrompt,
                4, // unet.in_channels
                _height / 8,
                _width / 8);
        }

        public Parameters GetParameters() {
            if (_prompt is not SingleInput) {
                throw new NotImplementedException("GetParameters not yet implemented for batch inputs.");
            }

            return new Parameters() {
                PackageVersion = System.Diagnostics.FileVersionInfo.GetVersionInfo(System.Reflection.Assembly.GetExecutingAssembly().Location).ProductVersion,
                Prompt = (_prompt as SingleInput).Text,
                Model = NameOrPath,
                NegativePrompt = _negativePrompt != null ? (_negativePrompt as SingleInput).Text : null,
                Steps = _steps,
                Sampler = Scheduler.GetType().Name,
                CfgScale = _guidanceScale,
                Seed = _seed,
                Width = _width,
                Height = _height,
                Eta = _eta
            };
        }

        public void Dispose() {
            TextEncoder?.Dispose();
            VaeDecoder?.Dispose();
            TextEncoder?.Dispose();
            Scheduler?.Dispose();
            Unet?.Dispose();
            _ops?.Dispose();
        }
    }
}
