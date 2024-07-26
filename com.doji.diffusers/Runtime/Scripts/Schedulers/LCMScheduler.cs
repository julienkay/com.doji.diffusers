using static Doji.AI.Diffusers.ArrayUtils;
using Unity.Sentis;
using System;
using UnityEngine;
using System.Collections.Generic;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// LCMScheduler extends the denoising procedure introduced in
    /// denoising diffusion probabilistic models (DDPMs) with
    /// non-Markovian guidance.
    /// </summary>
    public class LCMScheduler : SchedulerFloat, IDisposable {

        public float[] Betas { get; private set; }
        public float FinalAlphaCumprod { get; private set; }

        public bool CustomTimesteps { get; private set; }

        /// <summary>
        /// The index counter for current timestep. It will increae 1 after each scheduler step.
        /// </summary>
        protected int? StepIndex { get; set; }

        /// <summary>
        /// The index for the first timestep. It should be set from pipeline before the inference.
        /// </summary>
        protected int? BeginIndex { get; set; }

        public LCMScheduler(SchedulerConfig config, BackendType backend) : base(config, backend) {
            Config.NumTrainTimesteps ??= 1000;
            Config.BetaStart ??= 0.00085f;
            Config.BetaEnd ??= 0.012f;
            Config.BetaSchedule ??= Schedule.ScaledLinear;
            Config.TrainedBetas ??= null;
            Config.OriginalInferenceSteps ??= 50;
            Config.ClipSample ??= false;
            Config.ClipSampleRange ??= 1.0f;
            Config.SetAlphaToOne ??= true;
            Config.StepsOffset ??= 0;
            Config.PredictionType ??= Prediction.Epsilon;
            Config.Thresholding ??= false;
            Config.DynamicThresholdingRatio ??= 0.995f;
            Config.SampleMaxValue ??= 1.0f;
            Config.TimestepSpacing ??= Spacing.Leading;
            Config.TimestepScaling ??= 10.0f;
            Config.RescaleBetasZeroSnr ??= false;

            Initialize();
        }

        private void Initialize() {
            Betas = GetBetas();

            // Rescale for zero SNR
            if (RescaleBetasZeroSnr) {
                Betas = DDIMScheduler.RescaleZeroTerminalSnr(Betas);
            }

            float[] alphas = Sub(1f, Betas);
            AlphasCumprodF = alphas.CumProd();
            AlphasCumprod = new TensorFloat(new TensorShape(alphas.Length), AlphasCumprodF);

            // At every step in ddim, we are looking into the previous alphas_cumprod
            // For the final step, there is no previous alphas_cumprod because we are already at 0
            // `set_alpha_to_one` decides whether we set this parameter simply to one or
            // whether we use the final alpha of the "non-previous" one.
            FinalAlphaCumprod = SetAlphaToOne ? 1.0f : AlphasCumprodF[0];


            Timesteps = ArangeF(0, NumTrainTimesteps).Reverse();
            CustomTimesteps = false;

            StepIndex = null;
            BeginIndex = null;
        }

        public override void SetTimesteps(int numInferenceSteps) {
            const int strength = 1;
            int originalSteps = OriginalInferenceSteps;
            if (originalSteps > NumTrainTimesteps) {
                throw new ArgumentException(
                $"`original_steps`: {originalSteps} cannot be larger than `self.config.train_timesteps`: " +
                $" {NumTrainTimesteps} as the unet model trained with this scheduler can only handle " +
                $" maximal {NumTrainTimesteps} timesteps.");
            }

            // LCM Timesteps Setting
            // The skipping step parameter k from the paper.
            int k = NumTrainTimesteps / originalSteps;
            // LCM Training/Distillation Steps Schedule
            // Currently, only a linearly-spaced schedule is supported (same as in the LCM distillation scripts).
            int num = originalSteps * strength;
            float[] lcmOriginTimesteps = Linspace(1, num, num);

            // 2. Calculate the LCM inference timestep schedule.
            // TODO: 2.1 custom timestep schedules (not yet supported)

            // 2.2 Create the "standard" LCM inference timestep schedule.
            if (numInferenceSteps > NumTrainTimesteps) {
                throw new ArgumentException(
                    $"`num_inference_steps`: {numInferenceSteps} cannot be larger than `self.config.train_timesteps`: " +
                    $" {NumTrainTimesteps} as the unet model trained with this scheduler can only handle " +
                    $" maximal {NumTrainTimesteps} timesteps.");
            }

            int skippingStep = lcmOriginTimesteps.Length / numInferenceSteps;

            if (skippingStep < 1) {
                throw new ArgumentException(
                    $"The combination of `original_steps x strength`: {originalSteps} x {strength} is smaller than " +
                    $"`num_inference_steps`: {numInferenceSteps}. Make sure to either reduce `num_inference_steps` " +
                    $"to a value smaller than {num} or increase `strength` to a value higher " +
                    $"than {((float)numInferenceSteps / originalSteps)}."
                );
            }
            NumInferenceSteps = numInferenceSteps;

            if (numInferenceSteps > originalSteps) {
                throw new ArgumentException(
                    $"`num_inference_steps`: {numInferenceSteps} cannot be larger than `original_inference_steps`: " +
                    $" {originalSteps} because the final timestep schedule will be a subset of the " +
                    $" `original_inference_steps`-sized initial timestep schedule."
               );
            }

            // LCM Inference Steps Schedule
            lcmOriginTimesteps = lcmOriginTimesteps.Reverse();
            // Select (approximately) evenly spaced indices from lcm_origin_timesteps.
            float[] inferenceIndices;
            inferenceIndices = Linspace(0, lcmOriginTimesteps.Length, numInferenceSteps, endpoint: false);
            int[] inferenceIndicesI = inferenceIndices.Floor();

            Timesteps = lcmOriginTimesteps.Gather(inferenceIndicesI);

            StepIndex = null;
            BeginIndex = null;
        }

        public (float, float) GetScalingsForBoundaryConditionDiscrete(int timestep) {
            float sigma_data = 0.5f; // Default: 0.5
            float scaled_timestep = timestep * TimestepScaling;

            float c_skip = MathF.Pow(sigma_data, 2f) / (MathF.Pow(scaled_timestep, 2f) + MathF.Pow(sigma_data, 2f));
            float c_out = scaled_timestep / MathF.Sqrt(MathF.Pow(scaled_timestep, 2f) + MathF.Pow(sigma_data, 2f));
            return (c_skip, c_out);
        }

        public override SchedulerOutput Step(StepArgs args) {
            base.SetStepArgs(args);

            if (NumInferenceSteps == 0) {
                throw new ArgumentException("Number of inference steps is '0', you need to run 'SetTimesteps' after creating the scheduler");
            }

            if (StepIndex == null) {
                InitStepIndex(timestep);
            }

            // 1. get previous step value
            int prev_step_index = StepIndex.Value + 1;
            float prev_timestep;
            if (prev_step_index < Timesteps.Length) {
                prev_timestep = Timesteps[prev_step_index];
            } else {
                prev_timestep = timestep;
            }

            // 2. compute alphas, betas
            float alpha_prod_t = AlphasCumprod[(int)timestep];
            float alpha_prod_t_prev = prev_timestep >= 0 ? AlphasCumprod[(int)prev_timestep] : FinalAlphaCumprod;

            float beta_prod_t = 1 - alpha_prod_t;
            float beta_prod_t_prev = 1 - alpha_prod_t_prev;

            // 3. Get scalings for boundary conditions
            (float c_skip, float c_out) = GetScalingsForBoundaryConditionDiscrete((int)timestep);

            // 4. Compute the predicted original sample x_0 based on the model parameterization
            TensorFloat predOriginalSample;

            if (PredictionType == Prediction.Epsilon) { // noise-prediction
                var tmp = _ops.Sub(sample, _ops.Mul(MathF.Sqrt(beta_prod_t), modelOutput));
                predOriginalSample = _ops.Div(tmp, MathF.Sqrt(alpha_prod_t));
            } else if (PredictionType == Prediction.Sample) { // x-prediction
                predOriginalSample = modelOutput;
            } else if (PredictionType == Prediction.V_Prediction) { // v-prediction
                var tmp1 = _ops.Mul(MathF.Sqrt(alpha_prod_t), sample);
                var tmp2 = _ops.Mul(MathF.Sqrt(beta_prod_t), modelOutput);
                predOriginalSample = _ops.Sub(tmp1, tmp2);
            } else {
                throw new ArgumentException(
                    $"prediction_type given as {PredictionType} must be one of `epsilon`, `sample` or `v_prediction` for `LCMScheduler`."
                );
            }

            // 5. Clip or threshold "predicted x_0"
            if (Thresholding) {
                predOriginalSample = ThresholdSample(predOriginalSample);
            } else if (ClipSample) {
                predOriginalSample = _ops.Clip(predOriginalSample, - ClipSampleRange, ClipSampleRange);
            }

            // 6. Denoise model output using boundary conditions
            var tmp3 = _ops.Mul(c_out, predOriginalSample);
            var tmp4 = _ops.Mul(c_skip, sample);
            var denoised = _ops.Add(tmp3, tmp4);

            // 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
            // Noise is not used on the final timestep of the timestep schedule.
            // This also means that noise is not used for one-step sampling.
            TensorFloat prevSample;
            if (StepIndex != NumInferenceSteps - 1) {
                int seed = generator.Next();
                var noise = _ops.RandomNormal(modelOutput.shape, 0, 1f, seed);
                var tmp5 = _ops.Mul(MathF.Sqrt(alpha_prod_t_prev), denoised);
                var tmp6 = _ops.Mul(Mathf.Sqrt(beta_prod_t_prev), noise);
                prevSample = _ops.Add(tmp5, tmp6);
            } else {
                prevSample = denoised;
            }

            // upon completion increase step index by one
            StepIndex++;

            return new SchedulerOutput(prevSample: prevSample, predOriginalSample: denoised);
        }

        /// <summary>
        /// "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        /// prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        /// s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        /// pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        /// photorealism as well as better image-text alignment, especially when using very large guidance weights."
        /// </summary>
        private TensorFloat ThresholdSample(TensorFloat sample) {
            var shape = sample.shape;
            int batch_size = shape[0];
            int channels = shape[1];
            Debug.Assert(shape.rank == 4, "The ThresholdSample method assumes a rank of 4.");

            // Flatten sample for doing quantile calculation along each image
            sample = _ops.Reshape(sample, new TensorShape(batch_size, channels * shape[2] * shape[3]));
            var abs_sample = _ops.Abs(sample);  // "a certain percentile absolute pixel value"

            var s = _ops.Quantile(abs_sample, DynamicThresholdingRatio, dim: 1);
            s = _ops.Clip(s, min: 1, max: SampleMaxValue);  // When clamped to min=1, equivalent to standard clipping to [-1, 1]
            s.Reshape(s.shape.Unsqueeze(1));  // (batch_size, 1) because clamp will broadcast along dim=0

            var neg_s = _ops.Neg(s);
            var clamped = _ops.Clamp(sample, neg_s, s);  // "we threshold xt0 to the range [-s, s] and then divide by s"
            sample = _ops.Div(clamped, s);

            sample.Reshape(shape);
            return sample;
        }

        internal int IndexForTimestep(float timestep, float[] scheduleTimesteps = null) {
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


        private void InitStepIndex(float timestep) {
            if (BeginIndex == null) {
                StepIndex = IndexForTimestep(timestep);
            } else {
                StepIndex = BeginIndex;
            }
        }

        public static LCMScheduler FromConfig(SchedulerConfig cfg, BackendType b) => FromConfig<LCMScheduler>(cfg, b);
    }
}