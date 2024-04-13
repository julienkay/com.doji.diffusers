using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Async Stable Diffusion pipeline methods.
    /// </summary>
    public partial class StableDiffusionPipeline {

        public override async Task<TensorFloat> GenerateAsync(Parameters parameters) {
            InitGenerate(parameters);

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

            TensorFloat promptEmbeds = await EncodePromptAsync(prompt, numImagesPerPrompt, doClassifierFreeGuidance, negativePrompt);

            // get the initial random noise unless the user supplied it
            TensorShape latentsShape = GetLatentsShape();
            if (latents == null) {
                latents = _ops.RandomNormal(latentsShape, 0, 1, seed.Value);
            } else if (latents.shape != latentsShape) {
                throw new ArgumentException($"Unexpected latents shape, got {latents.shape}, expected {latentsShape}");
            }

            // set timesteps
            Scheduler.SetTimesteps(numInferenceSteps);

            if (Scheduler.InitNoiseSigma > 1.0f) {
                latents = _ops.Mul(Scheduler.InitNoiseSigma, latents);
            }

            int i = 0;
            foreach (float t in Scheduler) {
                // expand the latents if doing classifier free guidance
                TensorFloat latentModelInput = doClassifierFreeGuidance ? _ops.Concatenate(latents, latents, 0) : latents;
                latentModelInput = Scheduler.ScaleModelInput(latentModelInput, t);

                // predict the noise residual
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(batchSize), t);

                TensorFloat noisePred = await Unet.ExecuteAsync(latentModelInput, timestep, promptEmbeds);

                // perform guidance
                if (doClassifierFreeGuidance) {
                    (var noisePredUncond, var noisePredText) = _ops.SplitHalf(noisePred, axis: 0);
                    var tmp = _ops.Sub(noisePredText, noisePredUncond);
                    var tmp2 = _ops.Mul(guidanceScale, tmp);
                    noisePred = _ops.Add(noisePredUncond, tmp2);
                }

                // compute the previous noisy sample x_t -> x_t-1
                var stepArgs = new Scheduler.StepArgs(noisePred, t, latents, eta, generator: generator);
                var schedulerOutput = Scheduler.Step(stepArgs);
                latents = schedulerOutput.PrevSample;

                if (callback != null) {
                    callback.Invoke(i / Scheduler.Order, t, latents);
                }

                i++;
            }

            TensorFloat result = _ops.Div(latents, 0.18215f);

            // batch decode
            if (batchSize > 1) {
                throw new NotImplementedException();
            }

            TensorFloat outputImage = await VaeDecoder.ExecuteAsync(result);
            outputImage = ImageProcessor.PostProcess(outputImage, doDenormalize: true);

            return outputImage;
        }

        private async Task<TensorFloat> EncodePromptAsync(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            TensorFloat promptEmbeds = null,
            TensorFloat negativePromptEmbeds = null)
        {
            if (promptEmbeds == null) {
                var textInputs = Tokenizer.Encode(
                    text: prompt,
                    padding: Padding.MaxLength,
                    maxLength: Tokenizer.ModelMaxLength,
                    truncation: Truncation.LongestFirst
                ) as InputEncoding;
                int[] textInputIds = textInputs.InputIds.ToArray() ?? throw new Exception("Failed to get input ids from tokenizer.");
                int[] untruncatedIds = (Tokenizer.Encode(text: prompt, padding: Padding.Longest) as InputEncoding).InputIds.ToArray();

                if (untruncatedIds.Length >= textInputIds.Length && !textInputIds.ArrayEqual(untruncatedIds)) {
                    //TODO: support decoding tokens to text to be able to eventually display to user
                    UnityEngine.Debug.LogWarning("A part of your input was truncated because CLIP can only handle sequences up to " +
                    $"{Tokenizer.ModelMaxLength} tokens.");
                }

                using TensorInt textIdTensor = new TensorInt(new TensorShape(batchSize, textInputIds.Length), textInputIds);

                promptEmbeds = (await TextEncoder.ExecuteAsync(textIdTensor))[0] as TensorFloat;
            }

            promptEmbeds = _ops.Repeat(promptEmbeds, numImagesPerPrompt, axis: 0);

            // get unconditional embeddings for classifier free guidance
            if (doClassifierFreeGuidance && negativePromptEmbeds == null) {
                List<string> uncondTokens;
                if (negativePrompt == null) {
                    uncondTokens = Enumerable.Repeat("", batchSize).ToList();
                } else if (prompt.GetType() != negativePrompt.GetType()) {
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

                int maxLength = promptEmbeds.shape[1];
                var uncondInput = Tokenizer.Encode<BatchInput>(
                    text: uncondTokens,
                    padding: Padding.MaxLength,
                    maxLength: maxLength,
                    truncation: Truncation.LongestFirst
                ) as BatchEncoding;
                int[] uncondInputIds = uncondInput.InputIds as int[] ?? throw new Exception("Failed to get unconditioned input ids.");

                using TensorInt uncondIdTensor = new TensorInt(new TensorShape(batchSize, uncondInputIds.Length), uncondInputIds);

                promptEmbeds = _ops.Copy(promptEmbeds);  // "take ownership"
                negativePromptEmbeds = (await TextEncoder.ExecuteAsync(uncondIdTensor))[0] as TensorFloat;
            }

            if (doClassifierFreeGuidance) {
                negativePromptEmbeds = _ops.Repeat(negativePromptEmbeds, numImagesPerPrompt, axis: 0);

                // For classifier free guidance, we need to do two forward passes.
                // Here we concatenate the unconditional and text embeddings into a single batch
                // to avoid doing two forward passes
                TensorFloat combinedEmbeddings = _ops.Concatenate(negativePromptEmbeds, promptEmbeds, 0);

                return combinedEmbeddings;
            }

            return promptEmbeds;
        }
    }
}
