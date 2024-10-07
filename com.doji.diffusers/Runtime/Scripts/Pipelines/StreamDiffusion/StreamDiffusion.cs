using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public enum CfgType { None, Full, Self, Initialize }

    /// <summary>
    /// Experimental!!!
    /// </summary>
    public class StreamDiffusion : IDisposable {

        private int Width { get; set; }
        private int Height { get; set; }

        private int LatentWidth { get; set; }
        private int LatentHeight { get; set; }

        private int FrameBffSize { get; set; }
        private int NumDenoisingSteps { get; set; }

        private CfgType CfgType { get; set; }

        private int BatchSize { get; set; }
        private int TrtUnetBatchSize { get; set; }

        private List<int> TList { get; set; }

        private bool DoAddNoise { get; set; }
        private bool UseDenoisingBatch { get; set; }

        private bool SimilarImageFilter { get; set; }
        private SimilarImageFilter SimilarFilter { get; set; }
        private Tensor<float> PrevImageResult { get; set; }

        private DiffusionPipeline Pipe { get; set; }
        private VaeImageProcessor ImageProcessor { get; set; }

        private LCMScheduler Scheduler { get; set; }
        private TextEncoder TextEncoder { get; set; }
        private Unet Unet { get; set; }
        private VaeEncoder VaeEncoder { get; set; }
        private VaeDecoder VaeDecoder { get; set; }

        private Ops _ops { get; set; }

        private int InferenceTimeEma { get; set; }

        private float _guidanceScale;
        private float _delta;

        private Tensor<float> prompt_embeds;
        private Tensor<float> _initNoise;
        private Tensor<float> _stockNoise;
        private Tensor<float> c_skip;
        private float[] c_skip_list;
        private Tensor<float> c_out;
        private float[] c_out_list;
        private Tensor<float> alpha_prod_t_sqrt;
        private float[] alpha_prod_t_sqrt_list;
        private Tensor<float> beta_prod_t_sqrt;
        private float[] beta_prod_t_sqrt_list;
        private Tensor<int> sub_timesteps_tensor;
        private int[] sub_timesteps;
        private Tensor<float> x_t_latent_buffer;

        public StreamDiffusion(
            DiffusionPipeline pipe,
            List<int> tIndexList,
            int width = 512,
            int height = 512,
            bool doAddNoise = true,
            bool useDenoisingBatch = true,
            int frameBufferSize = 1,
            CfgType cfgType = CfgType.Self,
            BackendType backendType = BackendType.GPUCompute)
        {
            Height = height;
            Width = width;

            LatentHeight = height / pipe.VaeScaleFactor;
            LatentWidth = width / pipe.VaeScaleFactor;

            FrameBffSize = frameBufferSize;
            NumDenoisingSteps = tIndexList.Count;

            CfgType = cfgType;

            if (useDenoisingBatch) {
                BatchSize = NumDenoisingSteps * frameBufferSize;
                switch (cfgType) {
                    case CfgType.Initialize:
                        TrtUnetBatchSize = (NumDenoisingSteps + 1) * frameBufferSize;
                        break;
                    case CfgType.Full:
                        TrtUnetBatchSize = 2 * NumDenoisingSteps * frameBufferSize;
                        break;
                    default:
                        TrtUnetBatchSize = NumDenoisingSteps * frameBufferSize;
                        break;
                }
            } else {
                TrtUnetBatchSize = frameBufferSize;
                BatchSize = frameBufferSize;
            }

            TList = tIndexList;

            DoAddNoise = doAddNoise;
            UseDenoisingBatch = useDenoisingBatch;

            SimilarImageFilter = false;
            SimilarFilter = new SimilarImageFilter();
            PrevImageResult = null;

            Pipe = pipe;
            ImageProcessor = new VaeImageProcessor(vaeScaleFactor: pipe.VaeScaleFactor);

            Scheduler = LCMScheduler.FromConfig(pipe.Scheduler.Config, backendType);
            TextEncoder = pipe.TextEncoder;
            Unet = pipe.Unet;
            VaeEncoder = null; //TODO: either make pipe an Img2ImgPipeline, or think about making vae required in base class
            VaeDecoder = pipe.VaeDecoder;

            InferenceTimeEma = 0;

            _ops = new Ops(backendType);
        }

        public void Prepare(
            string prompt,
            string negative_prompt = null,
            int num_inference_steps = 50,
            float guidance_scale = 1.2f,
            float delta = 1.0f,
            uint? seed = null)
        {
            System.Random generator = new System.Random();
            seed ??= unchecked((uint)generator.Next());

            // initialize x_t_latent (it can be any random tensor)
            if (NumDenoisingSteps > 1) {
                x_t_latent_buffer = new Tensor<float>(
                    new TensorShape(
                        (NumDenoisingSteps - 1) * FrameBffSize,
                        4,
                        LatentHeight,
                        LatentWidth
                    )
                );
            } else {
                x_t_latent_buffer = null;
            }

            if (CfgType == CfgType.None) {
                guidance_scale = 1.0f;
            } else {
                _guidanceScale = guidance_scale;
            }
            _delta = delta;

            bool do_classifier_free_guidance = guidance_scale > 1.0f ? true : false;

            var embeddings = Pipe.EncodePrompt(
                prompt: prompt,
                numImagesPerPrompt: 1,
                doClassifierFreeGuidance: do_classifier_free_guidance,
                negativePrompt: negative_prompt
            );
            prompt_embeds = _ops.Repeat(embeddings.PromptEmbeds, BatchSize, axis: 0);

            Tensor<float> uncond_prompt_embeds = null;
            if (UseDenoisingBatch && CfgType == CfgType.Full) {
                uncond_prompt_embeds = _ops.Repeat(embeddings.NegativePromptEmbeds, BatchSize, axis: 0);
            } else if (CfgType == CfgType.Initialize) {
                uncond_prompt_embeds = _ops.Repeat(embeddings.NegativePromptEmbeds, FrameBffSize, axis: 0);
            }

            if (guidance_scale > 1.0f && (CfgType == CfgType.Initialize || CfgType == CfgType.Full)) {
                prompt_embeds = _ops.Concatenate(uncond_prompt_embeds, prompt_embeds, axis: 0);
            }

            Scheduler.SetTimesteps(num_inference_steps);
            var timesteps = Scheduler.Timesteps;

            // make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
            sub_timesteps = new int[num_inference_steps];
            for (int i = 0; i < TList.Count; i++) {
                int t = TList[i];
                sub_timesteps[i] = (int)timesteps[t];
            }

            var sub_timesteps_tensor = new Tensor<int>(new TensorShape(sub_timesteps.Length), sub_timesteps);
            int repeats = UseDenoisingBatch ? FrameBffSize : 1;
            this.sub_timesteps_tensor = _ops.RepeatInterleave(sub_timesteps_tensor, repeats, dim: 0);

            var latentsShape = new TensorShape(BatchSize, 4, LatentHeight, LatentWidth);
            _initNoise = _ops.RandomNormal(latentsShape, 0, 1, unchecked((int)seed));

            _stockNoise = new Tensor<float>(latentsShape);

            c_skip_list = new float[sub_timesteps.Length];
            c_out_list = new float[sub_timesteps.Length];
            for (int i = 0, timestep = sub_timesteps[i]; i < sub_timesteps.Length; i++) {
                (float c_skip, float c_out) = Scheduler.GetScalingsForBoundaryConditionDiscrete(timestep);
                c_skip_list[i] = c_skip;
                c_out_list[i] = c_out;
            }

            c_skip = new Tensor<float>(new TensorShape(c_skip_list.Length), c_skip_list);
            c_skip.Reshape(new TensorShape(c_skip_list.Length, 1, 1, 1));

            c_out = new Tensor<float>(new TensorShape(c_out_list.Length), c_out_list);
            c_out.Reshape(new TensorShape(c_out_list.Length, 1, 1, 1));

            alpha_prod_t_sqrt_list = new float[sub_timesteps.Length];
            beta_prod_t_sqrt_list = new float[sub_timesteps.Length];
            for (int i = 0, timestep = sub_timesteps[i]; i < sub_timesteps.Length; i++) {
                float alpha_prod_t_sqrt = MathF.Sqrt(Scheduler.AlphasCumprodF[timestep]);
                float beta_prod_t_sqrt = MathF.Sqrt(1f - Scheduler.AlphasCumprodF[timestep]);
                alpha_prod_t_sqrt_list[i] = alpha_prod_t_sqrt;
                beta_prod_t_sqrt_list[i] = beta_prod_t_sqrt;
            }
            alpha_prod_t_sqrt = new Tensor<float>(new TensorShape(alpha_prod_t_sqrt_list.Length), alpha_prod_t_sqrt_list);
            alpha_prod_t_sqrt.Reshape(new TensorShape(alpha_prod_t_sqrt_list.Length, 1, 1, 1));
            beta_prod_t_sqrt = new Tensor<float>(new TensorShape(beta_prod_t_sqrt_list.Length), beta_prod_t_sqrt_list);
            beta_prod_t_sqrt.Reshape(new TensorShape(beta_prod_t_sqrt_list.Length, 1, 1, 1));

            alpha_prod_t_sqrt = _ops.RepeatInterleave(alpha_prod_t_sqrt, repeats, dim: 0);
            beta_prod_t_sqrt = _ops.RepeatInterleave(beta_prod_t_sqrt, repeats, dim: 0);
        }

        private Tensor<float> AddNoise(Tensor<float> original_samples, Tensor<float> noise, int t_index) {
            var a = _ops.Mul(alpha_prod_t_sqrt_list[t_index], original_samples);
            var b = _ops.Mul(beta_prod_t_sqrt[t_index], noise);
            return _ops.Add(a, b);
        }

        private Tensor<float> scheduler_step_batch(
            Tensor<float> model_pred_batch,
            Tensor<float> x_t_latent_batch,
            int? idx = null)
        {
            // TODO: use t_list to select beta_prod_t_sqrt
            Tensor<float> F_theta;
            Tensor<float> denoised_batch;
            if (idx == null) {
                var tmp = _ops.Mul(beta_prod_t_sqrt, model_pred_batch);
                var tmp2 = _ops.Sub(x_t_latent_batch, tmp);
                F_theta = _ops.Div(tmp2, alpha_prod_t_sqrt);
                var tmp3 = _ops.Mul(c_out, F_theta);
                var tmp4 = _ops.Mul(c_skip, x_t_latent_batch);
                denoised_batch = _ops.Add(tmp3, tmp4);
            } else {
                var tmp = _ops.Mul(beta_prod_t_sqrt_list[idx.Value], model_pred_batch);
                var tmp2 = _ops.Sub(x_t_latent_batch, tmp);
                F_theta = _ops.Div(tmp2, alpha_prod_t_sqrt_list[idx.Value]);
                var tmp3 = _ops.Mul(c_out_list[idx.Value], F_theta);
                var tmp4 = _ops.Mul(c_skip_list[idx.Value], x_t_latent_batch);
                denoised_batch = _ops.Add(tmp3, tmp4);
            }
            return denoised_batch;
        }

        private (Tensor<float>, Tensor<float>) UnetStep(
            Tensor<float> x_t_latent,
            Tensor<int> t_list,
            int? idx = null)
        {
            Tensor<float> x_t_latent_plus_uc;
            if (_guidanceScale > 1.0f && CfgType == CfgType.Initialize) {
                x_t_latent_plus_uc = _ops.Concatenate(_ops.Split(x_t_latent, axis: 0, 0, 1), x_t_latent);
                t_list = _ops.Concatenate(_ops.Split(t_list, axis: 0, 0, 1), t_list);
            } else if (_guidanceScale > 1.0f && CfgType == CfgType.Full) {
                x_t_latent_plus_uc = _ops.Concatenate(x_t_latent, x_t_latent);
                t_list = _ops.Concatenate(t_list, t_list);
            } else {
                x_t_latent_plus_uc = x_t_latent;
            }

            Tensor<float> t = _ops.Cast(t_list);
            UnityEngine.Debug.Log(x_t_latent_plus_uc.shape);
            UnityEngine.Debug.Log(t.shape);
            UnityEngine.Debug.Log(prompt_embeds.shape);
            var model_pred = Unet.Execute(
                x_t_latent_plus_uc,
                t,
                encoderHiddenStates: prompt_embeds
            );

            Tensor<float> noise_pred_text;
            Tensor<float> noise_pred_uncond = null;
            if (_guidanceScale > 1.0f && CfgType == CfgType.Initialize) {
                noise_pred_text = _ops.Split(model_pred, axis: 0, start: 1, end: model_pred.shape[0]);
                _stockNoise = _ops.Concatenate(_ops.Split(model_pred, axis: 0, 0, 1), _ops.Split(_stockNoise, axis: 0, 1, _stockNoise.shape[0]));
            } else if (_guidanceScale > 1.0f && (CfgType == CfgType.Full)) {
                (noise_pred_uncond, noise_pred_text) = _ops.SplitHalf(model_pred);
            } else {
                noise_pred_text = model_pred;
            }
            if (_guidanceScale > 1.0f && (CfgType == CfgType.Self || CfgType == CfgType.Initialize)) {
                noise_pred_uncond = _ops.Mul(_stockNoise, _delta);
            }
            if (_guidanceScale > 1.0 && CfgType != CfgType.None) {
                var a = _ops.Add(noise_pred_uncond, _guidanceScale);
                var b = _ops.Sub(noise_pred_text, noise_pred_uncond);
                    model_pred = _ops.Mul(a, b);
            } else {
                model_pred = noise_pred_text;
            }

            // compute the previous noisy sample x_t -> x_t-1
            Tensor<float> denoised_batch;
            if (UseDenoisingBatch) {
                denoised_batch = scheduler_step_batch(model_pred, x_t_latent, idx);
                if (CfgType == CfgType.Self || CfgType == CfgType.Initialize) {
                    var scaled_noise = _ops.Mul(beta_prod_t_sqrt, _stockNoise);
                    var delta_x = scheduler_step_batch(model_pred, scaled_noise, idx);
                    var a = _ops.Split(alpha_prod_t_sqrt, axis: 0, start: 1, end: alpha_prod_t_sqrt.shape[0]);
                    TensorShape s = alpha_prod_t_sqrt.shape;
                    s[0] = 1;
                    using Tensor<float> b = new Tensor<float>(s, ArrayUtils.Full(s.length, 1f)); // torch.ones_like()
                    var alpha_next = _ops.Concatenate(a, b);
                    delta_x = _ops.Mul(alpha_next, delta_x);
                    var c = _ops.Split(beta_prod_t_sqrt, axis: 0, start: 1, end: beta_prod_t_sqrt.shape[0]);
                    s = beta_prod_t_sqrt.shape;
                    s[0] = 1;
                    using Tensor<float> d = new Tensor<float>(s, ArrayUtils.Full(s.length, 1f)); // torch.ones_like()
                    var beta_next = _ops.Concatenate(c, d);
                    delta_x = _ops.Div(delta_x, beta_next);
                    var e = _ops.Split(_initNoise, axis: 0, start: 1, end: _initNoise.shape[0]);
                    var f = _ops.Split(_stockNoise, axis: 0, start: 0, end: 1);
                    _initNoise = _ops.Concatenate(e, f);
                    _stockNoise = _ops.Add(_initNoise, delta_x);
                }
            } else {
                // denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
                denoised_batch = scheduler_step_batch(model_pred, x_t_latent, idx);
            }
            return (denoised_batch, model_pred);
        }

        private Tensor<float> EncodeImage(Tensor<float> image_tensors) {
            var img_latent = RetrieveLatents(VaeEncoder.Execute(image_tensors));
            img_latent = _ops.Mul(img_latent, VaeEncoder.Config.ScalingFactor.Value);
            Tensor<float> x_t_latent = AddNoise(img_latent, _initNoise, 0);
            return x_t_latent;
        }

        private Tensor<float> RetrieveLatents(Tensor<float> latentTensors) {
            return latentTensors;
        }

        private Tensor<float> decode_image(Tensor<float> x_0_pred_out) {
            var output_latent = VaeDecoder.Execute(_ops.Div(x_0_pred_out, VaeDecoder.Config.ScalingFactor.Value));
            return output_latent;
        }

        private Tensor<float> predict_x0_batch(Tensor<float> x_t_latent) {
            var prev_latent_batch = x_t_latent_buffer;

            Tensor<float> x_0_pred_out = null;
            if (UseDenoisingBatch) {
                var t_list = sub_timesteps_tensor;
                if (NumDenoisingSteps > 1) {
                    x_t_latent = _ops.Concatenate(x_t_latent, prev_latent_batch, axis: 0);
                    var a = _ops.Split(_initNoise, axis: 0, start: 0, end: 1);
                    var b = _ops.Split(_stockNoise, axis: 0, start: 1, end: _stockNoise.shape[0] - 1);
                    _stockNoise = _ops.Concatenate(a, b, axis: 0);

                }
                (Tensor<float> x_0_pred_batch, Tensor<float> model_pred) = UnetStep(x_t_latent, t_list);

                if (NumDenoisingSteps > 1) {
                    x_0_pred_out = _ops.Split(x_0_pred_batch, axis: 0, x_0_pred_batch.shape[0] - 1, x_0_pred_batch.shape[0]);
                    if (DoAddNoise) {
                        var a = _ops.Split(alpha_prod_t_sqrt, axis: 0, 1, alpha_prod_t_sqrt.shape[0]);
                        var b = _ops.Split(x_0_pred_batch, axis: 0, 0, x_0_pred_batch.shape[0] - 1);
                        var tmp = _ops.Mul(a, b);
                        var c = _ops.Split(beta_prod_t_sqrt, axis: 0, 1, beta_prod_t_sqrt.shape[0]);
                        var d = _ops.Split(_initNoise, axis: 0, 1, _initNoise.shape[0]);
                        var tmp2 = _ops.Mul(c, d);
                        x_t_latent_buffer = _ops.Add(tmp, tmp2);
                    } else {
                        var a = _ops.Split(alpha_prod_t_sqrt, axis: 0, 1, alpha_prod_t_sqrt.shape[0]);
                        var b = _ops.Split(x_0_pred_batch, axis: 0, 0, x_0_pred_batch.shape[0] - 1);
                        x_t_latent_buffer = _ops.Mul(a, b);
                    }
                } else {
                    x_0_pred_out = x_0_pred_batch;
                    x_t_latent_buffer = null;
                }
            } else {
                _initNoise = x_t_latent;
                for (int idx = 0; idx < sub_timesteps.Length; idx++) {
                    Tensor<int> t = new Tensor<int>(new TensorShape(), new[] { sub_timesteps[idx] });
                    t = _ops.Repeat(t, FrameBffSize, 0);
                    (Tensor<float> x_0_pred, Tensor<float> model_pred) = UnetStep(x_t_latent, t, idx);
                    if (idx < sub_timesteps_tensor.shape[0] - 1) {
                        if (DoAddNoise) {
                            var randn = _ops.RandomNormal(x_0_pred.shape, 0, 1, seed: 42);
                            var a = _ops.Mul(alpha_prod_t_sqrt_list[idx + 1], x_0_pred);
                            var b = _ops.Mul(beta_prod_t_sqrt_list[idx + 1], randn);
                            x_t_latent = _ops.Add(a, b);
                        } else {
                            x_t_latent = _ops.Mul(alpha_prod_t_sqrt_list[idx + 1], x_0_pred);
                        }
                    }
                    x_0_pred_out = x_0_pred;
                }
            }
            return x_0_pred_out;
        }

        public Tensor<float> Update(Tensor<float> x = null) {
            Tensor<float> x_t_latent;
            if (x != null) {
                x = ImageProcessor.PreProcess(x, Height, Width);
                if (SimilarImageFilter) {
                    x = SimilarFilter.Execute(x);
                    if (x is null) {
                        return PrevImageResult;
                    }
                }
                x_t_latent = EncodeImage(x);
            } else {
                // TODO: check the dimension of x_t_latent
                x_t_latent = _ops.RandomNormal(new TensorShape(1, 4, LatentHeight, LatentWidth), 0, 1, seed: new System.Random().Next());
            }

            var x_0_pred_out = predict_x0_batch(x_t_latent);
            var x_output = decode_image(x_0_pred_out);//.detach().clone();

            PrevImageResult = x_output;
            return x_output;
        }

        public void Dispose() {
            _stockNoise?.Dispose();
            c_skip?.Dispose();
            c_out?.Dispose();
            alpha_prod_t_sqrt?.Dispose();
            beta_prod_t_sqrt?.Dispose();
        }
    }
}