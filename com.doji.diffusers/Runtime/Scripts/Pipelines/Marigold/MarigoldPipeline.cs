using Doji.AI.Transformers;
using System;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Experimental!!!
    /// Pipeline for monocular depth estimation using Marigold
    /// <seealso href="https://marigoldmonodepth.github.io"/>
    /// </summary>
    public class MarigoldPipeline : IDisposable {

        public VaeEncoder VaeEncoder { get; protected set; }
        public VaeDecoder VaeDecoder { get; protected set; }
        public ClipTokenizer Tokenizer { get; protected set; }
        public TextEncoder TextEncoder { get; protected set; }
        public Scheduler Scheduler { get; protected set; }

        public Unet Unet { get; protected set; }

        /// <summary>
        /// RGB latent scale factor
        /// </summary>
        private const float RGB_SCALE = 0.18215f;

        /// <summary>
        /// depth latent scale factor
        /// </summary>
        private const float DEPTH_SCALE = 0.18215f;

        private int _steps;
        private int _inputHeight;
        private int _inputWidth;
        private int _batchSize;
        private uint? _seed;
        private System.Random _generator;

        private Ops _ops;
        private Tensor<float> _emptyTextEmbed;

        public MarigoldPipeline(
            VaeEncoder vaeEncoder,
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            BackendType backend)
        {
            VaeEncoder= vaeEncoder;
            VaeDecoder = vaeDecoder;
            Tokenizer = tokenizer;
            TextEncoder = textEncoder;
            Scheduler = scheduler;
            Unet = unet;
            _ops =  new Ops(backend);
        }


        public Tensor<float> Generate(
            Texture2D inputImage,
            int processingRes = 768,
            bool matchInputRes = true,
            int denoisingSteps = 10,
            int ensembleSize = 10,
            int batchSize = 0,
            string colorMap = "Spectral")
        {
            if (inputImage == null) {
                throw new ArgumentNullException(nameof(inputImage));
            }
            //TODO: do any image resizing + normalization

            using Tensor<float> input = TextureConverter.ToTensor(inputImage);
            var x = Generate(input, inputImage.width, inputImage.height, processingRes, matchInputRes, denoisingSteps, ensembleSize, batchSize, colorMap);

            throw new NotImplementedException();
        }


        /// <summary>
        /// Run depth estimation.
        /// </summary>
        /// <remarks>
        /// No support for batch generation / ensembling yet.
        /// </remarks>
        /// <param name="inputImage"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="processingRes"></param>
        /// <param name="matchInputRes"></param>
        /// <param name="denoisingSteps"></param>
        /// <param name="ensembleSize"></param>
        /// <param name="batchSize"></param>
        /// <param name="colorMap"></param>
        /// <returns></returns>
        public Tensor<float> Generate(
            Tensor<float> inputImage,
            int height,
            int width,
            int processingRes = 768,
            bool matchInputRes = true,
            int denoisingSteps = 10,
            int ensembleSize = 10,
            int batchSize = 0,
            string colorMap = "Spectral",
            uint? seed = null,
            Tensor<float> latents = null)
        {
            if (inputImage== null) {
                throw new ArgumentNullException(nameof(inputImage));
            }

            _steps = denoisingSteps;
            _inputHeight = height;
            _inputWidth = width;
            _generator = null;
            if (latents == null && _seed == null) {
                _generator = new System.Random();
                _seed = unchecked((uint)_generator.Next());
            }

            //TODO: check inputs
            //TODO: Resize image + make sure it's normalized
            Tensor<float> rgb = inputImage;

            // ----------------- Predicting depth -----------------
            // Batch repeated input image
            Tensor duplicatedRgb = _ops.Repeat(rgb, ensembleSize, -1); //TODO: check if this is the correct way to implement (torch.stack([rgb] * 10))?
            if (batchSize > 0) {
                _batchSize = batchSize;
            } else {
                int inputRes = Max(rgb.shape, 1);
                _batchSize = FindBatchSize(ensembleSize, inputRes);
            }

            //TODO: actually implement estimating in batches + ensembling

            return SingleInfer(rgb);
        }

        /// <summary>
        /// Perform an individual depth prediction without ensembling.
        /// </summary>
        private Tensor<float> SingleInfer(Tensor<float> rgb) {
            // set timesteps
            Scheduler.SetTimesteps(_steps);

            // encode image
            var rgbLatent = EncodeRGB(rgb);

            // initial depth map (noise)
            uint seed = unchecked((uint)new System.Random().Next());
            var latentsShape = new TensorShape(_batchSize, 4, _inputHeight, _inputWidth);
            var latents = _ops.RandomNormal(latentsShape, 0, 1, unchecked((int)seed));
            Tensor<float> depthLatents = latents;

            // batched empty text embedding
            if (_emptyTextEmbed == null) {
                EncodeEmptyText();
            }
            var batchEmptyTextEmbed = _emptyTextEmbed;
            // batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1))  # [B, 2, 1024]

            foreach (float t in Scheduler) {
                using Tensor timestep = Unet.CreateTimestep(new TensorShape(_batchSize), t);

                var unetInput = _ops.Concatenate(rgbLatent, depthLatents, 1); // this order is important
                // predict the noise residual
                var noisePred = Unet.Execute(unetInput, timestep, batchEmptyTextEmbed);

                // compute the previous noisy sample x_t -> x_t-1
                var stepArgs = new Scheduler.StepArgs(noisePred, t, depthLatents, generator: _generator);
                var schedulerOutput = Scheduler.Step(stepArgs);
                depthLatents = schedulerOutput.PrevSample;
            }

            var depth = DecodeDepth(depthLatents);

            // clip prediction
            depth = _ops.Clip(depth, -1.0f, 1.0f);
            // shift to [0, 1]
            depth = _ops.Mad(depth, 0.5f, 0.5f);

            return depth;
        }

        /// <summary>
        /// Encode text embedding for empty prompt.
        /// </summary>
        private void EncodeEmptyText() {
            string prompt = "";
            var textInputs = Tokenizer.Encode(
                prompt,
                padding: Padding.None,
                maxLength: Tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            );
            int[] inputIds = textInputs.InputIds.ToArray();
            using var textIdTensor = new Tensor<int>(new TensorShape(_batchSize, inputIds.Length), inputIds);
            _emptyTextEmbed = TextEncoder.Execute(textIdTensor)[0] as Tensor<float>;
        }

        private Tensor<float> EncodeRGB(Tensor<float> rgb) {
            var h = VaeEncoder.Execute(rgb);
            throw new NotImplementedException();
            /*
            moments = self.vae.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim = 1)
            # scale latent
            rgb_latent = mean * self.rgb_latent_scale_factor
            */
        }

        /// <summary>
        /// Decode depth latent into depth map.
        /// </summary>
        private Tensor<float> DecodeDepth(Tensor<float> depthLatents) {
            // scale latent
            depthLatents = _ops.Div(depthLatents, DEPTH_SCALE);
            // decode
            throw new NotImplementedException();

            /*
            z = self.vae.post_quant_conv(depth_latent)
            stacked = self.vae.decoder(z)
            # mean of output channels
            depth_mean = stacked.mean(dim = 1, keepdim = True)
            return depth_mean
            */
        }

        private static int Max(TensorShape shape, int startAxis) {
            int max = -1;
            for (int i = startAxis; i < shape.length; i++) {
                max = Math.Max(max, shape[i]);
            }
            return max;
        }

        private int FindBatchSize(int ensembleSize, int inputRes) {
            return 1;
            //TOOD: implement batches > 1
        }

        public void Dispose() {
            _emptyTextEmbed?.Dispose();
            _ops?.Dispose();
        }
    }
}