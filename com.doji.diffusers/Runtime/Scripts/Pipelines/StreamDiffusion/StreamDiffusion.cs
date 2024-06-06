using Doji.AI.Transformers;
using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public enum CfgType { None, Full, Self, Initialize }

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
        private TensorFloat PrevImageResult { get; set; }

        private DiffusionPipeline Pipe { get; set; }
        private VaeImageProcessor ImageProcessor { get; set; }

        private LCMScheduler Scheduler { get; set; }
        private TextEncoder TextEncoder { get; set; }
        private Unet Unet { get; set; }
        private VaeEncoder VaeEncoder { get; set; }

        private Ops _ops { get; set; }

        private int InferenceTimeEma { get; set; }

        private float _guidanceScale;
        private float _delta;

        private TensorFloat _initNoise;
        private TensorFloat _stockNoise;
        private TensorFloat c_skip;
        private TensorFloat c_out;
        private TensorFloat alpha_prod_t_sqrt;
        private TensorFloat beta_prod_t_sqrt;

        public StreamDiffusion(
            DiffusionPipeline pipe,
            List<int> tIndexList,
            int width = 512,
            int height = 512,
            bool doAddNoise = true,
            bool useDenoisingBatch = true,
            int frameBufferSize = 1,
            CfgType cfgType = CfgType.Self,
            BackendType backendType = BackendType.GPUCompute) {
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

            InferenceTimeEma = 0;

            _ops = new Ops(backendType);
        }

        private void Prepare(
            Input prompt,
            Input negative_prompt = null,
            int num_inference_steps = 50,
            float guidance_scale = 1.2f,
            float delta = 1.0f,
            uint? seed = null) {
            System.Random generator = new System.Random();
            seed ??= unchecked((uint)generator.Next());

            // initialize x_t_latent (it can be any random tensor)
            TensorFloat x_t_latent_buffer;
            if (NumDenoisingSteps > 1) {
                x_t_latent_buffer = TensorFloat.AllocZeros(
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
            var prompt_embeds = _ops.Repeat(embeddings.PromptEmbeds, BatchSize, axis: 0);

            TensorFloat uncond_prompt_embeds = null;
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
            int[] sub_timesteps = new int[num_inference_steps];
            for (int i = 0; i < TList.Count; i++) {
                int t = TList[i];
                sub_timesteps[i] = (int)timesteps[t];
            }

            var sub_timesteps_tensor = new TensorInt(new TensorShape(sub_timesteps.Length), sub_timesteps);
            int repeats = UseDenoisingBatch ? FrameBffSize : 1;
            sub_timesteps_tensor = _ops.RepeatInterleave(sub_timesteps_tensor, repeats, dim: 0);

            var latentsShape = new TensorShape(BatchSize, 4, LatentHeight, LatentWidth);
            _initNoise = _ops.RandomNormal(latentsShape, 0, 1, unchecked((int)seed));

            _stockNoise = TensorFloat.AllocZeros(latentsShape);

            float[] c_skip_list = new float[sub_timesteps.Length];
            float[] c_out_list = new float[sub_timesteps.Length];
            for (int i = 0, timestep = sub_timesteps[i]; i < sub_timesteps.Length; i++) {
                (float c_skip, float c_out) = Scheduler.GetScalingsForBoundaryConditionDiscrete(timestep);
                c_skip_list[i] = c_skip;
                c_out_list[i] = c_out;
            }

            c_skip = new TensorFloat(new TensorShape(c_skip_list.Length), c_skip_list);
            c_skip.Reshape(new TensorShape(c_skip_list.Length, 1, 1, 1));

            c_out = new TensorFloat(new TensorShape(c_out_list.Length), c_out_list);
            c_out.Reshape(new TensorShape(c_out_list.Length, 1, 1, 1));

            float[] alpha_prod_t_sqrt_list = new float[sub_timesteps.Length];
            float[] beta_prod_t_sqrt_list = new float[sub_timesteps.Length];
            for (int i = 0, timestep = sub_timesteps[i]; i < sub_timesteps.Length; i++) {
                float alpha_prod_t_sqrt = MathF.Sqrt(Scheduler.AlphasCumprodF[timestep]);
                float beta_prod_t_sqrt = MathF.Sqrt(1f - Scheduler.AlphasCumprodF[timestep]);
                alpha_prod_t_sqrt_list[i] = alpha_prod_t_sqrt;
                beta_prod_t_sqrt_list[i] = beta_prod_t_sqrt;
            }
            alpha_prod_t_sqrt = new TensorFloat(new TensorShape(alpha_prod_t_sqrt_list.Length), alpha_prod_t_sqrt_list);
            alpha_prod_t_sqrt.Reshape(new TensorShape(alpha_prod_t_sqrt_list.Length, 1, 1, 1));
            beta_prod_t_sqrt = new TensorFloat(new TensorShape(beta_prod_t_sqrt_list.Length), beta_prod_t_sqrt_list);
            beta_prod_t_sqrt.Reshape(new TensorShape(beta_prod_t_sqrt_list.Length, 1, 1, 1));

            alpha_prod_t_sqrt = _ops.RepeatInterleave(alpha_prod_t_sqrt, repeats, dim: 0);
            beta_prod_t_sqrt = _ops.RepeatInterleave(beta_prod_t_sqrt, repeats, dim: 0);
        }


        private TensorFloat AddNoise(TensorFloat original_samples, TensorFloat noise, int t_index) {
            /*noisy_samples = (
                self.alpha_prod_t_sqrt[t_index] * original_samples
                + self.beta_prod_t_sqrt[t_index] * noise
            )
            return noisy_samples*/
            throw new NotImplementedException();
        }

        private TensorFloat EncodeImage(TensorFloat image_tensors) {
            var img_latent = RetrieveLatents(VaeEncoder.Execute(image_tensors));
            img_latent = _ops.Mul(img_latent, VaeEncoder.Config.ScalingFactor.Value);
            TensorFloat x_t_latent = AddNoise(img_latent, _initNoise, 0);
            return x_t_latent;
        }

        private TensorFloat RetrieveLatents(TensorFloat latentTensors) {
            return latentTensors;
        }

        private TensorFloat decode_image(TensorFloat x_0_pred_out) {
            /*var output_latent = self.vae.decode(
                x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
            )[0]
            return output_latent*/
            throw new NotImplementedException();
        }

        private TensorFloat predict_x0_batch(TensorFloat x_t_latent) {
            throw new NotImplementedException();
        }

        private TensorFloat Update(TensorFloat x) {
            TensorFloat x_t_latent;
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
                x_t_latent = _ops.RandomNormal(new TensorShape(1, 4, LatentHeight, LatentWidth), 0, 1, seed: 42);
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