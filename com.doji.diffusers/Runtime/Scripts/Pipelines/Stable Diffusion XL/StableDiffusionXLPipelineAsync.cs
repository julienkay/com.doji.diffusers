using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Async Stable Diffusion XL Pipeline methods.
    /// </summary>
    public partial class StableDiffusionXLPipeline {

        public override async Task<TensorFloat> GenerateAsync(
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
            Action<int, float, TensorFloat> callback = null)
        {
            return await GenerateAsync(prompt, height, width, numInferenceSteps, guidanceScale, negativePrompt, numImagesPerPrompt, eta, seed, latents, callback);
        }

        public async Task<TensorFloat> GenerateAsync(
            Input prompt,
            int? height = 512,
            int? width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            Input negativePrompt = null,
            int numImagesPerPrompt = 1,
            float eta = 0.0f,
            uint? seed = null,
            TensorFloat latents = null,
            Action<int, float, TensorFloat> callback = null,
            float guidanceRescale = 0.0f,
            (int width, int height)? originalSize = null,
            (int x, int y) cropsCoordsTopLeft = default((int, int)),
            (int width, int height)? targetSize = null)
        {
            // 0. Default height and width to unet
            _height = height ?? (Unet.Config.SampleSize * VaeScaleFactor);
            _width = width ?? (Unet.Config.SampleSize * VaeScaleFactor);
            originalSize ??= (_height, _width);
            targetSize ??= (_height, _width);

            _prompt = prompt;
            _negativePrompt = negativePrompt;
            _steps = numInferenceSteps;
            _guidanceScale = guidanceScale;
            _numImagesPerPrompt = numImagesPerPrompt;
            _eta = eta;
            _seed = seed;
            _latents = latents;
            CheckInputs();

            // 2. Define call parameters
            if (prompt == null) {
                throw new ArgumentNullException(nameof(prompt));
            } else if (prompt is TextInput) {
                _batchSize = 1;
            } else if (prompt is BatchInput prompts) {
                _batchSize = prompts.Sequence.Count;
            } else {
                throw new ArgumentException($"Invalid prompt argument {nameof(prompt)}");
            }

            System.Random generator = null;
            if (latents == null && _seed == null) {
                generator = new System.Random();
                _seed = unchecked((uint)generator.Next());
            }

            bool doClassifierFreeGuidance = guidanceScale > 1.0f;

            // 3. Encode input prompt
            Embeddings promptEmbeds = await EncodePromptAsync(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);

            // 4. Prepare timesteps
            Scheduler.SetTimesteps(numInferenceSteps);

            // 5. Prepare latent variables
            PrepareLatents();

            // 7. Prepare added time ids & embeddings
            TensorFloat addTextEmbeds = promptEmbeds.PooledPromptEmbeds;
            float[] timeIds = GetTimeIds(originalSize.Value, cropsCoordsTopLeft, targetSize.Value);
            TensorFloat addTimeIds = new TensorFloat(new TensorShape(1, timeIds.Length), timeIds);
            if (doClassifierFreeGuidance) {
                promptEmbeds.PromptEmbeds = _ops.Concatenate(promptEmbeds.NegativePromptEmbeds, promptEmbeds.PromptEmbeds, axis: 0);
                addTextEmbeds = _ops.Concatenate(promptEmbeds.NegativePooledPromptEmbeds, addTextEmbeds, axis: 0);
                addTimeIds = _ops.Concatenate(addTimeIds, addTimeIds, axis: 0);
            }
            addTimeIds = _ops.Repeat(addTimeIds, _batchSize * _numImagesPerPrompt, axis: 0);

            // 8. Denoising loop
            int num_warmup_steps = Scheduler.TimestepsLength - numInferenceSteps * Scheduler.Order;
            int i = 0;
            foreach (float t in Scheduler) {
                // expand the latents if doing classifier free guidance
                TensorFloat latentModelInput = doClassifierFreeGuidance ? _ops.Concatenate(_latents, _latents, 0) : _latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(_batchSize), t);

                TensorFloat noisePred = await Unet.ExecuteAsync(
                    latentModelInput,
                    timestep,
                    promptEmbeds.PromptEmbeds,
                    addTextEmbeds,
                    addTimeIds
                );

                // perform guidance
                if (doClassifierFreeGuidance) {
                    (var noisePredUncond, var noisePredText) = _ops.SplitHalf(noisePred, axis: 0);
                    var tmp = _ops.Sub(noisePredText, noisePredUncond);
                    var tmp2 = _ops.Mul(guidanceScale, tmp);
                    noisePred = _ops.Add(noisePredUncond, tmp2);
                    if (Math.Abs(guidanceRescale) > 0.0f) {
                        // Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noisePred = RescaleNoiseCfg(noisePred, noisePredText, guidanceRescale);
                    }
                }

                // compute the previous noisy sample x_t -> x_t-1
                var stepArgs = new Scheduler.StepArgs(noisePred, t, _latents, eta, generator: generator);
                var schedulerOutput = Scheduler.Step(stepArgs);
                _latents = schedulerOutput.PrevSample;

                if (i == Scheduler.TimestepsLength - 1 || ((i + 1) > num_warmup_steps && (i + 1) % Scheduler.Order == 0)) {
                    int stepIdx = i / Scheduler.Order;
                    if (callback != null) {
                        callback.Invoke(i / Scheduler.Order, t, _latents);
                    }
                }

                i++;
            }

            TensorFloat result = _ops.Div(_latents, VaeDecoder.Config.ScalingFactor ?? 0.18215f);

            // batch decode
            if (_batchSize > 1) {
                throw new NotImplementedException();
            }

            TensorFloat image = await VaeDecoder.ExecuteAsync(result);

            return image;
        }

        private async Task<Embeddings> EncodePromptAsync(
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

                    using TensorInt textIdTensor = new TensorInt(new TensorShape(_batchSize, textInputIds.Length), textInputIds);

                    var _promptEmbeds = await textEncoder.ExecuteAsync(textIdTensor);

                    pooledPromptEmbeds = _promptEmbeds[0] as TensorFloat;
                    promptEmbeds = _promptEmbeds[-2] as TensorFloat;

                    // copy prompt embeds to avoid having to call TakeOwnership and track tensor to Dispose()
                    promptEmbeds = _ops.Copy(promptEmbeds);
                    promptEmbeds = _ops.Repeat(promptEmbeds, numImagesPerPrompt, axis: 0);

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
                    uncondTokens = Enumerable.Repeat((negativePrompt as SingleInput).Text, _batchSize).ToList();
                } else if (_batchSize != (negativePrompt as BatchInput).Sequence.Count) {
                    throw new ArgumentException($"`negativePrompt`: {negativePrompt} has batch size {(negativePrompt as BatchInput).Sequence.Count}, " +
                        $"but `prompt`: {prompt} has batch size {_batchSize}. Please make sure that passed `negativePrompt` matches " +
                        $"the batch size of `prompt`.");
                } else {
                    uncondTokens = (negativePrompt as BatchInput).Sequence as List<string>;
                }

                _negativePromptEmbedsList.Clear();
                foreach (var (tokenizer, textEncoder) in Encoders) {
                    int maxLength = promptEmbeds.shape[1];
                    var uncondInput = tokenizer.Encode<BatchInput>(
                        text: uncondTokens,
                        padding: Padding.MaxLength,
                        maxLength: maxLength,
                        truncation: Truncation.LongestFirst
                    ) as BatchEncoding;
                    int[] uncondInputIds = uncondInput.InputIds as int[];

                    using TensorInt uncondIdTensor = new TensorInt(new TensorShape(_batchSize, uncondInputIds.Length), uncondInputIds);

                    var _negativePromptEmbeds = textEncoder.Execute(uncondIdTensor);

                    negativePooledPromptEmbeds = _negativePromptEmbeds[0] as TensorFloat;
                    negativePromptEmbeds = _negativePromptEmbeds[-2] as TensorFloat;
                    negativePromptEmbeds = _ops.Copy(negativePromptEmbeds);

                    // duplicate unconditional embeddings for each generation per prompt
                    negativePromptEmbeds = _ops.Repeat(negativePromptEmbeds, numImagesPerPrompt, axis: 0);

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
    }
}