using static Doji.AI.ArrayUtils;
using System;
using Unity.InferenceEngine;
using System.Collections.Generic;

namespace Doji.AI.Diffusers {

    public abstract class EulerScheduler : SchedulerFloat {

        public float[] Betas { get; private set; }
        internal float[] Sigmas { get; set; }
        public bool IsScaleInputCalled { get; private set; }

        public override float InitNoiseSigma {
            get {
                // TODO: can we calculate this once after calling SetTimesteps() and cache it?
                // standard deviation of the initial noise distribution
                if (TimestepSpacing == Spacing.Linspace || TimestepSpacing == Spacing.Trailing) {
                    return Sigmas.Max();
                }
                return MathF.Pow(MathF.Pow(Sigmas.Max(), 2f) + 1, 0.5f);
            }
        }

        /// <summary>
        /// The index counter for current timestep. It will increae 1 after each scheduler step.
        /// </summary>
        protected int? StepIndex { get; set; }

        /// <summary>
        /// The index for the first timestep. It should be set from pipeline before the inference.
        /// </summary>
        protected int? BeginIndex { get; set; }

        public EulerScheduler(SchedulerConfig config, BackendType backend = BackendType.GPUCompute) : base(config, backend) { }

        protected void Initialize() {
            Betas = GetBetas();

            if (RescaleBetasZeroSnr) {
                Betas = DDIMScheduler.RescaleZeroTerminalSnr(Betas);
            }

            float[] alphas = Sub(1f, Betas);
            AlphasCumprodF = alphas.CumProd();
            AlphasCumprod = new Tensor<float>(new TensorShape(alphas.Length), AlphasCumprodF);

            if (RescaleBetasZeroSnr) {
                // Close to 0 without being 0 so first sigma is not inf
                // FP16 smallest positive subnormal works well here
                //AlphasCumprod[^1] = MathF.Pow(2f, -24f);
                throw new NotImplementedException();
            }

            float[] tmp1 = Sub(1f, AlphasCumprodF);
            float[] tmp2 = tmp1.Div(AlphasCumprodF);
            Sigmas = tmp2.Pow(0.5f).Reverse();
            Timesteps = Linspace(0f, NumTrainTimesteps - 1, NumTrainTimesteps).Reverse();

            // setable values
            NumInferenceSteps = 0;

            Sigmas = Sigmas.Concatenate(0);

            IsScaleInputCalled = false;
            StepIndex = null;
            BeginIndex = null;
        }

        /// <summary>
        /// Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        /// </summary>
        /// <inheritdoc/>
        public override Tensor<float> ScaleModelInput(Tensor<float> sample, float timestep) {
            if (StepIndex == null) {
                InitStepIndex(timestep);
            }

            float sigma = Sigmas[StepIndex.Value];
            sample = Ops.Div(sample, MathF.Pow((MathF.Pow(sigma, 2f) + 1f), 0.5f));

            IsScaleInputCalled = true;
            return sample;
        }

        /// <inheritdoc/>
        public override void SetTimesteps(int numInferenceSteps) {
            NumInferenceSteps = numInferenceSteps;

            // "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if (TimestepSpacing == Spacing.Linspace) {
                Timesteps = GetTimeStepsLinspaceF().Reverse();
            } else if (TimestepSpacing == Spacing.Leading) {
                Timesteps = GetTimeStepsLeadingF().Reverse();
            } else if (TimestepSpacing == Spacing.Trailing) {
                Timesteps = GetTimeStepsTrailingF().Reverse();
            } else {
                throw new ArgumentException($"{TimestepSpacing} is not supported. Please choose one of {string.Join(", ", Enum.GetNames(typeof(Spacing)))}.");
            }

            float[] tmp1 = Sub(1f, AlphasCumprodF);
            float[] tmp2 = tmp1.Div(AlphasCumprodF);
            Sigmas = tmp2.Pow(0.5f);
            //var tmp1 = _ops.Sub(1f, AlphasCumprod);
            //var tmp2 = _ops.Div(tmp1, AlphasCumprod);
            //using Tensor<float> pow = new Tensor<float>(0.5f);
            //var sigmas = _ops.Pow(tmp2, pow);
        }

        public override Tensor<float> AddNoise(Tensor<float> originalSamples, Tensor<float> noise, Tensor<float> timesteps) {
            float[] scheduleTimesteps = Timesteps;

            int[] stepIndices;
            // BeginIndex is null when pipeline does not implement SetBeginIndex
            if (BeginIndex == null) {
                float[] timestepsF = timesteps.DownloadToArray();
                stepIndices = new int[timestepsF.Length];
                for (int i = 0; i < timestepsF.Length; i++) {
                    float t = timestepsF[i];
                    stepIndices[i] = (IndexForTimestep(t, scheduleTimesteps));
                }
            } else if (StepIndex != null) {
                // add_noise is called after first denoising step (for inpainting)
                stepIndices = new int[timesteps.shape[0]];
                for (int i = 0; i < timesteps.shape[0]; i++) {
                    stepIndices[i] = StepIndex.Value;
                }
            } else {
                // add noise is called before first denoising step to create inital latent(img2img)
                stepIndices = new int[timesteps.shape[0]];
                for (int i = 0; i < timesteps.shape[0]; i++) {
                    stepIndices[i] = BeginIndex.Value;
                }
            }

            using Tensor<int> indices = new Tensor<int>(new TensorShape(stepIndices.Length), stepIndices);
            using Tensor<float> sigmas = new Tensor<float>(new TensorShape(Sigmas.Length), Sigmas);
            var sigma = Ops.GatherElements(sigmas, indices, 0);
            while (sigma.shape.rank < originalSamples.shape.rank) {
                sigma.Reshape(sigma.shape.Unsqueeze(-1)); // unsqueeze
            }
            var noisySamples = Ops.Add(originalSamples, Ops.Mul(noise, sigma));
            return noisySamples;
        }

        private int IndexForTimestep(float timestep, float[] scheduleTimesteps = null) {
            scheduleTimesteps ??= Timesteps;

            List<int> indices = new List<int>();
            for (int i = 0; i < scheduleTimesteps.Length; i++) {
                if (scheduleTimesteps[i] == timestep) {
                    indices.Add(i);
                }
            }

            // The sigma index that is taken for the **very** first `step`
            // is always the second index (or the last index if there is only 1)
            // This way we can ensure we don't accidentally skip a sigma in
            // case we start in the middle of the denoising schedule (e.g. for image-to-image)
            int pos = indices.Count > 1 ? 1 : 0;

            return indices[pos];
        }

        protected void InitStepIndex(float timestep) {
            if (BeginIndex == null) {
                StepIndex = IndexForTimestep(timestep);
            } else {
                StepIndex = BeginIndex;
            }
        }

        public override void Dispose() {
            AlphasCumprod?.Dispose();
            base.Dispose();
        }
    }
}