using Doji.AI.Transformers;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public enum CfgType { None, Full, Self, Initialize }

    public class StreamDiffusion {

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
        private object PrevImageResult { get; set; }

        private DiffusionPipeline Pipe { get; set; }
        private VaeImageProcessor ImageProcessor { get; set; }

        private LCMScheduler Scheduler { get; set; }
        private TextEncoder TextEncoder { get; set; }
        private Unet Unet { get; set; }
        private Ops _ops { get; set; }

        private int InferenceTimeEma { get; set; }

        private float _guidanceScale;
        private float _delta;

        private TensorFloat _initNoise;
        private TensorFloat _stockNoise;

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
            sub_timesteps_tensor = _ops.RepeatInterleave(sub_timesteps_tensor, repeats, dim = 0);

            var latentsShape = new TensorShape(BatchSize, 4, LatentHeight, LatentWidth);
            _initNoise = _ops.RandomNormal(latentsShape, 0, 1, unchecked((int)seed));

            _stockNoise = TensorFloat.AllocZeros(latentsShape);

            List<float> c_skip_list = new List<float>();
            List<float> c_out_list = new List<float>();
            foreach (int timestep in sub_timesteps) {
                (float c_skip, float c_out) = Scheduler.GetScalingsForBoundaryConditionDiscrete(timestep);
                c_skip_list.Add(c_skip);
                c_out_list.Add(c_out);
            }

            c_skip = (
                torch.stack(c_skip_list)
                .view(len(t_list), 1, 1, 1)
                .to(dtype = dtype, device = device)
            )
            c_out = (
                torch.stack(c_out_list)
                .view(len(t_list), 1, 1, 1)
                .to(dtype = dtype, device = device)
            )

            alpha_prod_t_sqrt_list = []
            beta_prod_t_sqrt_list = []
            for timestep in sub_timesteps:
                alpha_prod_t_sqrt = scheduler.alphas_cumprod[timestep].sqrt()
                beta_prod_t_sqrt = (1 - scheduler.alphas_cumprod[timestep]).sqrt()
                alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
                beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
            alpha_prod_t_sqrt = (
                torch.stack(alpha_prod_t_sqrt_list)
                .view(len(t_list), 1, 1, 1)
                .to(dtype = dtype, device = device)
            )
            beta_prod_t_sqrt = (
                torch.stack(beta_prod_t_sqrt_list)
                .view(len(t_list), 1, 1, 1)
                .to(dtype = dtype, device = device)
            )
            alpha_prod_t_sqrt = torch.repeat_interleave(
                alpha_prod_t_sqrt,
                repeats = frame_bff_size if use_denoising_batch else 1,
                dim = 0,
            )
            beta_prod_t_sqrt = torch.repeat_interleave(
                beta_prod_t_sqrt,
                repeats = frame_bff_size if use_denoising_batch else 1,
                dim = 0,
            )*/
        }

    }
}