using Doji.AI.Transformers;
using System;
using System.Linq;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

namespace Doji.AI.Diffusers {

    public enum ResampleMethod { Nearest, NearestExact, Bilinear, Bicubic, Area }

    /// <summary>
    /// Struct with arguments for precise ensembling control.
    /// </summary>
    public struct EnsemblingOptions {

        public enum ReductionMethod { Median, Mean }

        /// <summary>
        /// Defines the ensembling function applied in every pixel location.
        /// </summary>
        public ReductionMethod Reduction;

        /// <summary>
        /// Strength of the regularizer that pulls the aligned predictions to the unit range from 0 to 1.
        /// </summary>
        public float RegularizerStrength;

        /// <summary>
        /// Maximum number of the alignment solver steps.
        /// </summary>
        public int MaxIter;

        /// <summary>
        /// Alignment solver tolerance. The solver stops when the tolerance is reached.
        /// </summary>
        public float Tol;

        /// <summary>
        /// Resolution at which the alignment is performed. <see langword="null"/> matches the <see cref="Parameters.ProcessingResolution"/>.
        /// </summary>
        public int? MaxRes;

        public EnsemblingOptions(
            ReductionMethod reduction = ReductionMethod.Median,
            float regularizerStrength = 0.02f,
            int maxIter = 2,
            float tol = 1e-3f,
            int? maxRes = null)
        {
            Reduction = reduction;
            RegularizerStrength = regularizerStrength;
            MaxIter = maxIter;
            Tol = tol;
            MaxRes = maxRes;
        }
    }

    /// <summary>
    /// Pipeline for monocular depth estimation using Marigold
    /// <seealso href="https://marigoldmonodepth.github.io"/>
    /// </summary>
    public partial class MarigoldDepthPipeline : DiffusionPipeline, IDisposable {

        private MarigoldPipelineConfig _config;
        public override PipelineConfig Config {
            get { return _config; }
            protected set { _config = value as MarigoldPipelineConfig; }
        }

        public enum PredictionType { Depth, Normals }

        public VaeEncoder VaeEncoder { get; protected set; }

#pragma warning disable IDE1006 // Naming Styles

        /* Parameters accessors for convenience
         * only valid inside Generate() calls */

        protected int ensembleSize { get => _parameters.EnsembleSize.Value; set => _parameters.EnsembleSize = value; }
        protected int? processingResolution { get => _parameters.ProcessingResolution; set => _parameters.ProcessingResolution = value; }
        protected bool matchInputResolution { get => _parameters.MatchInputResolution.Value; set => _parameters.MatchInputResolution = value; }
        protected ResampleMethod resampleMethodInput { get => _parameters.ResampleMethodInput.Value; set => _parameters.ResampleMethodInput = value; }
        protected ResampleMethod resampleMethod { get => _parameters.ResampleMethodOutput.Value; set => _parameters.ResampleMethodOutput = value; }
        protected EnsemblingOptions ensemblingOptions { get => _parameters.EnsemblingOptions.Value; set => _parameters.EnsemblingOptions = value; }
        protected bool outputUncertainty { get => _parameters.OutputUncertainty.Value; set => _parameters.OutputUncertainty = value; }

#pragma warning restore IDE1006

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
        private int? _seed;
        private System.Random _generator;

        private Tensor<float> _emptyTextEmbedding;
        private new MarigoldImageProcessor ImageProcessor { get; set; }

        public MarigoldDepthPipeline(
            VaeEncoder vaeEncoder,
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            BackendType backend) : base(vaeDecoder, textEncoder, tokenizer, scheduler, unet, backend)
        {
            VaeEncoder = vaeEncoder;
            ImageProcessor = new MarigoldImageProcessor(VaeScaleFactor);
        }

        public override Parameters GetDefaultParameters() {
            return new Parameters() {
                NumInferenceSteps = _config.DefaultDenoisingSteps,
                EnsembleSize = 1,
                ProcessingResolution = _config.DefaultProcessingResolution,
                MatchInputResolution = true,
                ResampleMethodInput = ResampleMethod.Bilinear,
                ResampleMethodOutput = ResampleMethod.Bilinear,
                EnsemblingOptions = new EnsemblingOptions(),
                OutputUncertainty = false
            };
        }

        public Tensor<float> Generate(
            Tensor<float> image,
            int? numInferenceSteps = null,
            int ensembleSize = 1,
            int? processingResolution = null,
            bool matchInputResolution = true,
            ResampleMethod resampleMethodInput = ResampleMethod.Bilinear,
            ResampleMethod resampleMethodOutput = ResampleMethod.Bilinear,
            EnsemblingOptions? ensemblingOptions = null,
            bool outputUncertainty = false)
        {
            Parameters p = new Parameters() {
                Image = image,
                NumInferenceSteps = numInferenceSteps,
                EnsembleSize = ensembleSize,
                ProcessingResolution = processingResolution,
                MatchInputResolution = matchInputResolution,
                ResampleMethodInput = resampleMethodInput,
                ResampleMethodOutput = resampleMethodOutput,
                EnsemblingOptions = ensemblingOptions ?? new EnsemblingOptions(),
                OutputUncertainty = outputUncertainty
            };
            return Generate(p);
        }

        public override Tensor<float> Generate(Parameters parameters) {
            InitGenerate(parameters);

            // 2. Prepare empty text conditioning.
            if (_emptyTextEmbedding == null) {
                EncodeEmptyText();
            }

            // 3. Preprocess input images. This function loads input image or images of compatible dimensions `(H, W)`,
            // optionally downsamples them to the `processing_resolution` `(PH, PW)`, where
            // `max(PH, PW) == processing_resolution`, and pads the dimensions to `(PPH, PPW)` such that these values are
            // divisible by the latent space downscaling factor (typically 8 in Stable Diffusion). The default value `None`
            // of `processing_resolution` resolves to the optimal value from the model config. It is a recommended mode of
            // operation and leads to the most reasonable results. Using the native image resolution or any other processing
            // resolution can lead to loss of either fine details or global context in the output predictions.
            (var processed, var padding, var original_resolution) = ImageProcessor.Preprocess(
                image, processingResolution, resampleMethodInput
            ); // [N,3,PPH,PPW]
            image = processed;

            // 4. Encode input image into latent space. At this step, each of the `N` input images is represented with `E`
            // ensemble members. Each ensemble member is an independent diffused prediction, just initialized independently.
            // Latents of each such predictions across all input images and all ensemble members are represented in the
            // `pred_latent` variable. The variable `image_latent` is of the same shape: it contains each input image encoded
            // into latent space and replicated `E` times. The latents can be either generated (see `generator` to ensure
            // reproducibility), or passed explicitly via the `latents` argument. The latter can be set outside the pipeline
            // code. For example, in the Marigold-LCM video processing demo, the latents initialization of a frame is taken
            // as a convex combination of the latents output of the pipeline for the previous frame and a newly-sampled
            // noise. This behavior can be achieved by setting the `output_latent` argument to `True`. The latent space
            // dimensions are `(h, w)`. Encoding into latent space happens in batches of size `batch_size`.
            // Model invocation: self.vae.encoder.
            (var image_latent, var pred_latent) = PrepareLatents(); // [N*E,4,h,w], [N*E,4,h,w]

            var batchEmptyTextEmbedding = batchSize > 1 ?
                _ops.Repeat(_emptyTextEmbedding, batchSize, 0) :
                _emptyTextEmbedding;

            // 5. Process the denoising loop. All `N * E` latents are processed sequentially in batches of size `batch_size`.
            // The unet model takes concatenated latent spaces of the input image and the predicted modality as an input, and
            // outputs noise for the predicted modality's latent space. The number of denoising diffusion steps is defined by
            // `num_inference_steps`. It is either set directly, or resolves to the optimal value specific to the loaded
            // model.
            // Model invocation: self.unet.
            Tensor<float> predLatents;

            return null;
        }

        private (Tensor<float>, Tensor<float>) PrepareLatents() {
            Tensor<float> initLatents = VaeEncoder.Execute(image);
            Tensor<float> imageLatents = _ops.Mul(initLatents, VaeDecoder.Config.ScalingFactor.Value);
            imageLatents = _ops.RepeatInterleave(imageLatents, ensembleSize, dim: 0); // [N*E,4,h,w]

            var predLatent = latents;
            predLatent ??= _ops.RandomNormal(imageLatents.shape, 0, 1, seed.Value); // [N*E,4,h,w]
            return (imageLatents, predLatent);
        }

        protected override void CheckInputs() {
            base.CheckInputs();
            if (ensembleSize > 1) {
                throw new ArgumentException($"{nameof(ensembleSize)} must be positive, was {ensembleSize}.");
            }
            if (ensembleSize == 2) {
                Debug.LogWarning("{nameof(ensembleSize) == 2 results are similar to no ensembling." +
                    "Consider increasing the value to at least 3.");
            }
            if (ensembleSize > 1 && _config.ScaleInvariant  || _config.ShiftInvariant) {
                throw new NotImplementedException("Ensembling not yet implemented.");
            }
            if (ensembleSize == 1 && outputUncertainty) {
                throw new ArgumentException($"Computing uncertainty by setting '{nameof(outputUncertainty)}=True` also requires setting `{nameof(ensembleSize)}` greater than 1.");
            }
            if (processingResolution == null) {
                throw new ArgumentException($"`{nameof(processingResolution)}` is not specified and could not be resolved from the model config.");
            }
            if (processingResolution.Value < 0) {
                throw new ArgumentException($"`{nameof(processingResolution)}` must be non-negative: 0 for native resolution, or any positive value for downsampled processing.");
            }
            if (processingResolution.Value % VaeScaleFactor != 0) {
                throw new ArgumentException($"`{nameof(processingResolution)}` must be a multiple of {VaeScaleFactor}.");
            }
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
            int? seed = null,
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
                _seed = _generator.Next();
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
            int seed = new System.Random().Next();
            var latentsShape = new TensorShape(_batchSize, 4, _inputHeight, _inputWidth);
            var latents = _ops.RandomNormal(latentsShape, 0, 1, seed);
            Tensor<float> depthLatents = latents;

            // batched empty text embedding
            if (_emptyTextEmbedding == null) {
                EncodeEmptyText();
            }
            var batchEmptyTextEmbed = _emptyTextEmbedding;
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
            _emptyTextEmbedding = TextEncoder.Execute(textIdTensor)[0] as Tensor<float>;
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
        public override void Dispose() {
            base.Dispose();
            VaeEncoder?.Dispose();
            _emptyTextEmbedding?.Dispose();
        }

        internal override Embeddings EncodePrompt(Transformers.Input prompt, int numImagesPerPrompt, bool doClassifierFreeGuidance, Transformers.Input negativePrompt = null, Tensor<float> promptEmbeds = null, Tensor<float> negativePromptEmbeds = null, Tensor<float> pooledPromptEmbeds = null, Tensor<float> negativePooledPromptEmbeds = null) {
            throw new NotImplementedException();
        }
    }
}