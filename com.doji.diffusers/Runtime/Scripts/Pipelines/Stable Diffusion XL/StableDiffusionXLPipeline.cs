using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using Unity.Sentis;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Stable Diffusion XL Pipeline 
    /// </summary>
    /// <remarks>
    /// pipeline_stable_diffusion_xl.py from huggingface/optimum
    /// </remarks>
    public partial class StableDiffusionXLPipeline : DiffusionPipeline, ITxt2ImgPipeline, IDisposable {

        public ClipTokenizer Tokenizer2 { get; private set; }
        public TextEncoder TextEncoder2 { get; private set; }

        public List<(ClipTokenizer Tokenizer, TextEncoder TextEncoder)> Encoders { get; set; }
        
        public int VaeScaleFactor { get; set; }

        private List<TensorFloat> _promptEmbedsList = new List<TensorFloat>();
        private List<TensorFloat> _negativePromptEmbedsList = new List<TensorFloat>();

        /// <summary>
        /// Initializes a new Stable Diffusion XL pipeline.
        /// </summary>
        public StableDiffusionXLPipeline(
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            TextEncoder textEncoder2,
            ClipTokenizer tokenizer2,
            BackendType backend) : base(backend)
        {
            VaeDecoder = vaeDecoder;
            Tokenizer = tokenizer;
            Tokenizer2 = tokenizer2;
            TextEncoder = textEncoder;
            TextEncoder2 = textEncoder2;
            Scheduler = scheduler;
            Unet = unet;
            Encoders = Tokenizer != null && TextEncoder != null
                ? new() { (Tokenizer, TextEncoder), (Tokenizer2, TextEncoder2) }
                : new() { (Tokenizer2, TextEncoder2) };

            if (VaeDecoder.Config.BlockOutChannels != null) {
                VaeScaleFactor = 1 << (VaeDecoder.Config.BlockOutChannels.Length - 1);
            } else {
                VaeScaleFactor = 8;
            }
        }

        public override Parameters GetDefaultParameters() {
            return new Parameters() {
                Height = null,
                Width = null,
                NumInferenceSteps = 50,
                GuidanceScale = 5.0f,
                NegativePrompt = null,
                NumImagesPerPrompt = 1,
                Eta = 0.0f,
                Seed = null,
                Latents = null,
                Callback = null,
                GuidanceRescale = 0.0f,
                OriginalSize = null,
                CropsCoordsTopLeft = (0, 0),
                TargetSize = null,
            };
        }

        public override TensorFloat Generate(Parameters parameters) {
            Profiler.BeginSample($"{GetType().Name}.Generate");

            SetParameterDefaults(parameters);

            // Default height and width to unet
            _parameters.Height ??= (Unet.Config.SampleSize * VaeScaleFactor);
            _parameters.Width ??= (Unet.Config.SampleSize * VaeScaleFactor);
            originalSize ??= (height, width);
            targetSize ??= (height, width);

            CheckInputs();

            // 2. Define call parameters
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            } else if (prompt is TextInput) {
                batchSize = 1;
            } else if (prompt is BatchInput prompts) {
                batchSize = prompts.Sequence.Count;
            } else {
                throw new ArgumentException($"Invalid prompt argument {nameof(prompt)}");
            }

            System.Random generator = null;
            if (latents == null && seed == null) {
                generator = new System.Random();
                seed = unchecked((uint)generator.Next());
            }

            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            // 3. Encode input prompt
            Profiler.BeginSample("Encode Prompt(s)");
            Embeddings promptEmbeds = EncodePrompt(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);
            Profiler.EndSample();

            // 4. Prepare timesteps
            Profiler.BeginSample($"{Scheduler.GetType().Name}.SetTimesteps");
            Scheduler.SetTimesteps(numInferenceSteps);
            Profiler.EndSample();

            // 5. Prepare latent variables
            PrepareLatents();

            // 7. Prepare added time ids & embeddings
            TensorFloat addTextEmbeds = promptEmbeds.PooledPromptEmbeds;
            float[] timeIds = GetTimeIds(originalSize.Value, cropsCoordsTopLeft.Value, targetSize.Value);
            using TensorFloat initAddTimeIds = new TensorFloat(new TensorShape(1, timeIds.Length), timeIds);
            TensorFloat addTimeIds = initAddTimeIds;

            if (doClassifierFreeGuidance) {
                promptEmbeds.PromptEmbeds = _ops.Concatenate(promptEmbeds.NegativePromptEmbeds, promptEmbeds.PromptEmbeds, axis: 0);
                addTextEmbeds = _ops.Concatenate(promptEmbeds.NegativePooledPromptEmbeds, addTextEmbeds, axis: 0);
                addTimeIds = _ops.Concatenate(addTimeIds, addTimeIds, axis: 0);
            }
            addTimeIds = _ops.Repeat(addTimeIds, batchSize * numImagesPerPrompt, axis: 0);

            // 8. Denoising loop
            Profiler.BeginSample($"Denoising Loop");
            int numWarmupSteps = Scheduler.TimestepsLength - numInferenceSteps * Scheduler.Order;
            int i = 0;
            foreach (float t in Scheduler) {
                // expand the latents if doing classifier free guidance
                TensorFloat latentModelInput = doClassifierFreeGuidance ? _ops.Concatenate(latents, latents, 0) : latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                Profiler.BeginSample("Prepare Timestep Tensor");
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(batchSize), t);
                Profiler.EndSample();

                Profiler.BeginSample("Execute Unet");
                TensorFloat noisePred = Unet.Execute(
                    latentModelInput,
                    timestep,
                    promptEmbeds.PromptEmbeds,
                    addTextEmbeds,
                    addTimeIds
                );
                Profiler.EndSample();

                // perform guidance
                if (doClassifierFreeGuidance) {
                    Profiler.BeginSample("Extend Predicted Noise For Classifier-Free Guidance");
                    (var noisePredUncond, var noisePredText) = _ops.SplitHalf(noisePred, axis: 0);
                    var tmp = _ops.Sub(noisePredText, noisePredUncond);
                    var tmp2 = _ops.Mul(guidanceScale, tmp);
                    noisePred = _ops.Add(noisePredUncond, tmp2);
                    if (Math.Abs(guidanceRescale) > 0.0f) {
                        // Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noisePred = RescaleNoiseCfg(noisePred, noisePredText, guidanceRescale);
                    }
                    Profiler.EndSample();
                }

                // compute the previous noisy sample x_t -> x_t-1
                Profiler.BeginSample($"{Scheduler.GetType().Name}.Step");
                var stepArgs = new Scheduler.StepArgs(noisePred, t, latents, eta, generator: generator);
                var schedulerOutput = Scheduler.Step(stepArgs);
                latents = schedulerOutput.PrevSample;
                Profiler.EndSample();

                if (i == Scheduler.TimestepsLength - 1 || ((i + 1) > numWarmupSteps && (i + 1) % Scheduler.Order == 0)) {
                    int stepIdx = i / Scheduler.Order;
                    if (callback != null) {
                        Profiler.BeginSample($"{GetType()} Callback");
                        callback.Invoke(i / Scheduler.Order, t, latents);
                        Profiler.EndSample();
                    }
                }

                i++;
            }
            Profiler.EndSample();

            Profiler.BeginSample($"Scale Latents");
            TensorFloat result = _ops.Div(latents, VaeDecoder.Config.ScalingFactor ?? 0.18215f);
            Profiler.EndSample();

            // batch decode
            if (batchSize > 1) {
                throw new NotImplementedException();
            }

            Profiler.BeginSample($"VaeDecoder Decode Image");
            TensorFloat image = VaeDecoder.Execute(result);
            Profiler.EndSample();

            Profiler.BeginSample($"PostProcess Image");
            image = ImageProcessor.PostProcess(image);
            Profiler.EndSample();

            Profiler.EndSample();
            return image;
        }

        private Embeddings EncodePrompt(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null,
            TensorFloat pooledPromptEmbeds = null,
            TensorFloat negativePooledPromptEmbeds = null)
        {
            if (promptEmbeds == null) {
                _promptEmbedsList.Clear(); 

                foreach (var (tokenizer, textEncoder) in Encoders) {
                    Profiler.BeginSample("CLIPTokenizer Encode Input");
                    var textInputs = tokenizer.Encode(
                        text: prompt,
                        padding: Padding.MaxLength,
                        maxLength: tokenizer.ModelMaxLength,
                        truncation: Truncation.LongestFirst
                    ) as InputEncoding;
                    int[] textInputIds = textInputs.InputIds.ToArray();
                    int[] untruncatedIds = (tokenizer.Encode(text: prompt, padding: Padding.Longest) as InputEncoding).InputIds.ToArray();

                    if (untruncatedIds.Length >= textInputIds.Length && !textInputIds.ArrayEqual(untruncatedIds)) {
                        //TODO: support decoding tokens to text to be able to eventually display to user
                        UnityEngine.Debug.LogWarning("A part of your input was truncated because CLIP can only handle sequences up to " +
                        $"{tokenizer.ModelMaxLength} tokens.");
                    }
                    Profiler.EndSample();

                    Profiler.BeginSample("Prepare Text ID Tensor");
                    using TensorInt textIdTensor = new TensorInt(new TensorShape(batchSize, textInputIds.Length), textInputIds);
                    Profiler.EndSample();

                    Profiler.BeginSample("Execute TextEncoder");
                    var _promptEmbeds = textEncoder.Execute(textIdTensor);
                    Profiler.EndSample();

                    pooledPromptEmbeds = _promptEmbeds[0] as TensorFloat;
                    promptEmbeds = _promptEmbeds[-2] as TensorFloat;

                    // copy prompt embeds to avoid having to call TakeOwnership and track tensor to Dispose()
                    promptEmbeds = _ops.Copy(promptEmbeds);

                    Profiler.BeginSample($"Process Input for {numImagesPerPrompt} images per prompt.");
                    promptEmbeds = _ops.Repeat(promptEmbeds, numImagesPerPrompt, axis: 0);
                    Profiler.EndSample();

                    _promptEmbedsList.Add(promptEmbeds);
                }
                promptEmbeds = _ops.Concatenate(_promptEmbedsList, -1);
            }

            // get unconditional embeddings for classifier free guidance
            bool zeroOutNegativePrompt = negativePrompt is null && Config.ForceZerosForEmptyPrompt;
            if (doClassifierFreeGuidance && negativePromptEmbeds is null && zeroOutNegativePrompt) {
                using var zeros = TensorFloat.Zeros(promptEmbeds.shape);
                negativePromptEmbeds = zeros;
                using var zerosP = TensorFloat.Zeros(pooledPromptEmbeds.shape);
                negativePooledPromptEmbeds = zerosP;
            } else if (doClassifierFreeGuidance && negativePromptEmbeds is null) {
                negativePrompt = negativePrompt ?? "";
                List<string> uncondTokens;
                if (prompt is not null && prompt.GetType() != negativePrompt.GetType()) {
                    throw new ArgumentException($"`negativePrompt` should be the same type as `prompt`, but got {negativePrompt.GetType()} != {prompt.GetType()}.");
                } else if (negativePrompt is SingleInput) {
                    uncondTokens = Enumerable.Repeat((negativePrompt as SingleInput).Text, batchSize).ToList();
                } else if (batchSize != (negativePrompt as BatchInput).Sequence.Count) {
                    throw new ArgumentException($"`negativePrompt`: {negativePrompt} has batch size {(negativePrompt as BatchInput).Sequence.Count}, " +
                        $"but `prompt`: {prompt} has batch size {batchSize}. Please make sure that passed `negativePrompt` matches " +
                        $"the batch size of `prompt`.");
                } else {
                    uncondTokens = (negativePrompt as BatchInput).Sequence as List<string>;
                }

                _negativePromptEmbedsList.Clear();
                foreach (var (tokenizer, textEncoder) in Encoders) {
                    Profiler.BeginSample("CLIPTokenizer Encode Unconditioned Input");
                    int maxLength = promptEmbeds.shape[1];
                    var uncondInput = tokenizer.Encode<BatchInput>(
                        text: uncondTokens,
                        padding: Padding.MaxLength,
                        maxLength: maxLength,
                        truncation: Truncation.LongestFirst
                    ) as BatchEncoding;
                    int[] uncondInputIds = uncondInput.InputIds as int[];
                    Profiler.EndSample();

                    Profiler.BeginSample("Prepare Unconditioned Text ID Tensor");
                    using TensorInt uncondIdTensor = new TensorInt(new TensorShape(batchSize, uncondInputIds.Length), uncondInputIds);
                    Profiler.EndSample();

                    Profiler.BeginSample("Execute TextEncoder For Unconditioned Input");
                    var _negativePromptEmbeds = textEncoder.Execute(uncondIdTensor);
                    Profiler.EndSample();

                    negativePooledPromptEmbeds = _negativePromptEmbeds[0] as TensorFloat;
                    negativePromptEmbeds = _negativePromptEmbeds[-2] as TensorFloat;
                    negativePromptEmbeds = _ops.Copy(negativePromptEmbeds);

                    // duplicate unconditional embeddings for each generation per prompt
                    Profiler.BeginSample($"Process Unconditional Input for {numImagesPerPrompt} images per prompt.");
                    negativePromptEmbeds = _ops.Repeat(negativePromptEmbeds, numImagesPerPrompt, axis: 0);
                    Profiler.EndSample();

                    // For classifier free guidance, we need to do two forward passes.
                    // Here we concatenate the unconditional and text embeddings into a single batch
                    // to avoid doing two forward passes
                    _negativePromptEmbedsList.Add(negativePromptEmbeds);
                }
                negativePromptEmbeds = _ops.Concatenate(_negativePromptEmbedsList, -1);
            }

            pooledPromptEmbeds = _ops.Repeat(pooledPromptEmbeds, numImagesPerPrompt, axis: 0);
            negativePooledPromptEmbeds = _ops.Repeat(negativePooledPromptEmbeds, numImagesPerPrompt, axis: 0);

            return new Embeddings() {
                PromptEmbeds = promptEmbeds,
                NegativePromptEmbeds = negativePromptEmbeds,
                PooledPromptEmbeds = pooledPromptEmbeds,
                NegativePooledPromptEmbeds = negativePooledPromptEmbeds
            };
        }

        private void PrepareLatents() {
            var shape = new TensorShape(
                batchSize * numImagesPerPrompt,
                Unet.Config.InChannels ?? 4,
                height / 8,
                width / 8
            );

            if (latents == null) {
                latents = _ops.RandomNormal(shape, 0, 1, seed);
            } else if (latents.shape != shape) {
                throw new ArgumentException($"Unexpected latents shape, got {latents.shape}, expected {shape}");
            }
            
            // scale the initial noise by the standard deviation required by the scheduler
            if (Math.Abs(Scheduler.InitNoiseSigma - 1.0f) > 0.00001f) {
                latents = _ops.Mul(Scheduler.InitNoiseSigma, latents);
            }
        }

        private float[] GetTimeIds((int, int) a, (int, int) b, (int, int) c) {
            return new float[] {
                a.Item1,
                a.Item2,
                b.Item1,
                b.Item2,
                c.Item1,
                c.Item2
            };
        }

        /// <summary>
        /// Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of <see href="https://arxiv.org/pdf/2305.08891.pdf">
        /// Common Diffusion Noise Schedules and Sample Steps are Flawed</see>. See Section 3.4
        /// </summary>
        private TensorFloat RescaleNoiseCfg(TensorFloat noiseCfg, TensorFloat noise_pred_text, float guidanceRescale = 0.0f) {
            throw new NotImplementedException("guidanceRescale > 0.0f not supported yet.");
            /*TensorFloat std_text = np.std(noise_pred_text, axis = tuple(range(1, noise_pred_text.ndim)), keepdims = True);
            TensorFloat std_cfg = np.std(noiseCfg, axis = tuple(range(1, noiseCfg.ndim)), keepdims = True);
            // rescale the results from guidance (fixes overexposure)
            TensorFloat noisePredRescaled = noiseCfg * (std_text / std_cfg);
            // mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
            noiseCfg = guidanceRescale * noisePredRescaled + (1f - guidanceRescale) * noiseCfg;
            return noiseCfg;*/
        }

        public override void Dispose() {
            base.Dispose();
            TextEncoder2?.Dispose();
        }

        private struct Embeddings {
            public TensorFloat PromptEmbeds { get; set; }
            public TensorFloat NegativePromptEmbeds { get; set; }
            public TensorFloat PooledPromptEmbeds { get; set; }
            public TensorFloat NegativePooledPromptEmbeds { get; set; }
        }
    }
}