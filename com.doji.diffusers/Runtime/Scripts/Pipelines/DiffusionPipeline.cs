using Doji.AI.Transformers;
using System;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {


    public delegate void PipelineCallback(int step, float timestep, Tensor<float> latents);

    public abstract partial class DiffusionPipeline : IDiffusionPipeline, IDisposable {

        public DiffusionModel ModelInfo { get; protected set; }
        public PipelineConfig Config { get; protected set; }

        public VaeDecoder VaeDecoder { get; protected set; }
        public ClipTokenizer Tokenizer { get; protected set; }
        public TextEncoder TextEncoder { get; protected set; }
        public Scheduler Scheduler { get; protected set; }
        public Unet Unet { get; protected set; }

        public VaeImageProcessor ImageProcessor { get; protected set; }

        public int VaeScaleFactor { get; set; }

        internal Ops _ops;

        protected Parameters _parameters;

#pragma warning disable IDE1006 // Naming Styles

        /* Parameters accessors for convenience
         * only valid inside Generate() calls */

        protected Input prompt { get => _parameters.Prompt; set => _parameters.Prompt = value; }
        protected int height { get => _parameters.Height.Value; set => _parameters.Height = value; }
        protected int width { get => _parameters.Width.Value; set => _parameters.Width = value; }
        protected int numInferenceSteps { get => _parameters.NumInferenceSteps.Value; set => _parameters.NumInferenceSteps = value; }
        protected float guidanceScale { get => _parameters.GuidanceScale.Value; set => _parameters.GuidanceScale = value; }
        protected Input negativePrompt { get => _parameters.NegativePrompt; set => _parameters.NegativePrompt = value; }
        protected int numImagesPerPrompt { get => _parameters.NumImagesPerPrompt.Value; set => _parameters.NumImagesPerPrompt = value; }
        protected float eta { get => _parameters.Eta.Value; set => _parameters.Eta = value; }
        protected uint? seed { get => _parameters.Seed; set => _parameters.Seed = value; }
        protected Tensor<float> latents { get => _parameters.Latents; set => _parameters.Latents = value; }
        protected float guidanceRescale { get => _parameters.GuidanceRescale.Value; set => _parameters.GuidanceRescale = value; }
        protected PipelineCallback callback { get => _parameters.Callback; set => _parameters.Callback = value; }
        protected Tensor<float> image { get => _parameters.Image; set => _parameters.Image = value; }
        protected float strength { get => _parameters.Strength.Value; set => _parameters.Strength = value; }
        protected float controlnetConditioningScale { get => _parameters.ControlnetConditioningScale.Value; set => _parameters.ControlnetConditioningScale = value; }
        protected bool guessMode { get => _parameters.GuessMode.Value; set => _parameters.GuessMode = value; }
        protected float controlGuidanceStart { get => _parameters.ControlGuidanceStart.Value; set => _parameters.ControlGuidanceStart = value; }
        protected float controlGuidanceEnd { get => _parameters.ControlGuidanceEnd.Value; set => _parameters.ControlGuidanceEnd = value; }
        protected (int width, int height)? originalSize { get => _parameters.OriginalSize; set => _parameters.OriginalSize = value; }
        protected (int x, int y)? cropsCoordsTopLeft { get => _parameters.CropsCoordsTopLeft; set => _parameters.CropsCoordsTopLeft = value; }
        protected (int width, int height)? targetSize { get => _parameters.TargetSize; set => _parameters.TargetSize = value; }
        protected float aestheticScore { get => _parameters.AestheticScore.Value; set => _parameters.AestheticScore = value; }
        protected float negativeAestheticScore { get => _parameters.NegativeAestheticScore.Value; set => _parameters.NegativeAestheticScore = value; }
        protected int batchSize { get; set; } = 1;

#pragma warning restore IDE1006

        /// <summary>
        /// Base constructor for a diffusion pipeline.
        /// To load a pretrained model, use <see cref="FromPretrained(DiffusionModel, BackendType)"/>
        /// </summary>
        public DiffusionPipeline(
            VaeDecoder vaeDecoder,
            TextEncoder textEncoder,
            ClipTokenizer tokenizer,
            Scheduler scheduler,
            Unet unet,
            BackendType backend)
        {
            VaeDecoder = vaeDecoder;
            TextEncoder = textEncoder;
            Tokenizer = tokenizer;
            Scheduler = scheduler;
            Unet = unet;

            // TODO: When casting between pipeline types, we might want to reuse ops and image processor as well
            _ops = Scheduler.Ops = new Ops(backend);
            if (VaeDecoder.Config.BlockOutChannels != null) {
                VaeScaleFactor = 1 << (VaeDecoder.Config.BlockOutChannels.Length - 1);
            } else {
                VaeScaleFactor = 8;
            }
            ImageProcessor = new VaeImageProcessor(vaeScaleFactor: VaeScaleFactor, backend: backend);
        }

        /// <summary>
        /// Create a pipeline reusing the components of the given pipeline.
        /// </summary>
        public DiffusionPipeline(DiffusionPipeline pipe) : this(pipe.VaeDecoder, pipe.TextEncoder, pipe.Tokenizer, pipe.Scheduler, pipe.Unet, pipe._ops.BackendType) {
            ModelInfo = pipe.ModelInfo;
            Config = pipe.Config;
        }

        /// <summary>
        /// Applies default values for the specific pipeline.
        /// </summary>
        /// <param name="parameters">The parameters that were passed to the Generate() method.</param>
        protected void SetParameterDefaults(Parameters parameters) {
            // set the parameters that were passed by the user
            _parameters = parameters;

            // then make sure that all parameters that were not specified,
            // have the default value defined by the pipeline
            Parameters defaults = GetDefaultParameters();
            _parameters.Height ??= defaults.Height;
            _parameters.Width ??= defaults.Width;
            _parameters.NumInferenceSteps ??= defaults.NumInferenceSteps;
            _parameters.GuidanceScale ??= defaults.GuidanceScale;
            _parameters.NegativePrompt ??= defaults.NegativePrompt;
            _parameters.NumImagesPerPrompt ??= defaults.NumImagesPerPrompt;
            _parameters.Eta ??= defaults.Eta;
            _parameters.Seed ??= defaults.Seed;
            _parameters.Latents ??= defaults.Latents;
            _parameters.Callback ??= defaults.Callback;
            _parameters.Image ??= defaults.Image;
            _parameters.Strength ??= defaults.Strength;
            _parameters.GuidanceRescale ??= defaults.GuidanceRescale;
            _parameters.ControlnetConditioningScale ??= defaults.ControlnetConditioningScale;
            _parameters.GuessMode ??= defaults.GuessMode;
            _parameters.ControlGuidanceStart ??= defaults.ControlGuidanceStart;
            _parameters.ControlGuidanceEnd ??= defaults.ControlGuidanceEnd;
            _parameters.OriginalSize ??= defaults.OriginalSize;
            _parameters.CropsCoordsTopLeft ??= defaults.CropsCoordsTopLeft;
            _parameters.TargetSize ??= defaults.TargetSize;
            _parameters.AestheticScore ??= defaults.AestheticScore;
            _parameters.NegativeAestheticScore ??= defaults.NegativeAestheticScore;
        }

        public abstract Parameters GetDefaultParameters();

        protected void CheckInputs() {
            if (this is not IImg2ImgPipeline && (height % 8 != 0 || width % 8 != 0)) {
                throw new ArgumentException($"`height` and `width` have to be divisible by 8 but are {height} and {width}.");
            }
            if (numImagesPerPrompt > 1) {
                throw new ArgumentException($"More than one image per prompt not supported yet. `numImagesPerPrompt` was {numImagesPerPrompt}.");
            }
            if (latents != null && seed != null) {
                throw new ArgumentException($"Both a seed and pre-generated noise has been passed. Please use either one or the other.");
            }
            if (this is ITxt2ImgPipeline && prompt == null) {
                throw new ArgumentException("Please provide a 'prompt' parameter to generate images.");
            }
            if ((this is IImg2ImgPipeline || this is IControlnetPipeline) && image == null) {
                throw new ArgumentException($"Please provide an 'image' parameter to generate images using a {GetType()}.");
            }
        }

        public Metadata GetMetadata() {
            if (prompt is not SingleInput) {
                throw new NotImplementedException("GetParameters not yet implemented for batch inputs.");
            }

            return new Metadata() {
                Model = ModelInfo.ModelId,
                Pipeline = GetType().Name,
                Sampler = Scheduler.GetType().Name,
                Parameters = _parameters,
            };
        }

        protected void InitGenerate(Parameters parameters) {
            _ops.FlushTensors();
            ImageProcessor._ops.FlushTensors();
            Scheduler.Ops.FlushTensors();
            SetParameterDefaults(parameters);
            CheckInputs();
        }

        /// <summary>
        /// Execute the pipeline to generate images.
        /// </summary>
        /// <param name="parameters">the parameters used to generate the image</param>
        /// <returns>the resulting image tensor</returns>
        public abstract Tensor<float> Generate(Parameters parameters);

        // TODO: this should be moved to ITxt2Img pipeline, because these defaults only
        // make sense for that (e.g. no image parameters). Maybe 'FromPretrained' should
        // return loaded pipeline as Itxt2Img then.
        /// <summary>
        /// Execute the pipeline to generate images.
        /// </summary>
        /// <remarks>
        /// This is an overload for the most common generation parameters for convenience.
        /// For more control and advanced pipeline usage, pass parameters via the
        /// <see cref="Generate(Parameters)"/> method instead.
        /// </remarks>
        public Tensor<float> Generate(
            string prompt,
            int? width = null,
            int? height = null,
            int? numInferenceSteps = null,
            float? guidanceScale = null,
            string negativePrompt = null,
            uint? seed = null)
        {
            Parameters parameters = new Parameters() {
                Prompt = prompt,
                Width = width,
                Height = height,
                NumInferenceSteps = numInferenceSteps,
                GuidanceScale = guidanceScale,
                NegativePrompt = negativePrompt,
                Seed = seed,
            };
            return Generate(parameters);
        }

        /// <summary>
        /// Execute the pipeline to generate images asynchronously.
        /// </summary>
        /// <remarks>
        /// This is an overload for the most common generation parameters for convenience.
        /// For more control and advanced pipeline usage, pass parameters via the
        /// <see cref="GenerateAsync(Parameters)"/> method instead.
        /// </remarks>
        public Task<Tensor<float>> GenerateAsync(
            string prompt,
            int? width = null,
            int? height = null,
            int? numInferenceSteps = null,
            float? guidanceScale = null,
            string negativePrompt = null,
            uint? seed = null)
        {
            Parameters parameters = new Parameters() {
                Prompt = prompt,
                Width = width,
                Height = height,
                NumInferenceSteps = numInferenceSteps,
                GuidanceScale = guidanceScale,
                NegativePrompt = negativePrompt,
                Seed = seed,
            };
            return GenerateAsync(parameters);
        }

        /// <summary>
        /// Execute the pipeline asynchronously.
        /// </summary>
        public virtual Task<Tensor<float>> GenerateAsync(Parameters parameters) {
            throw new NotImplementedException($"{nameof(GenerateAsync)} not implemented for {GetType().Name}.");
        }

        internal abstract Embeddings EncodePrompt(
            Input prompt,
            int numImagesPerPrompt,
            bool doClassifierFreeGuidance,
            Input negativePrompt = null,
            Tensor<float> promptEmbeds = null,
            Tensor<float> negativePromptEmbeds = null,
            Tensor<float> pooledPromptEmbeds = null,
            Tensor<float> negativePooledPromptEmbeds = null
        );

        public virtual void Dispose() {
            VaeDecoder?.Dispose();
            TextEncoder?.Dispose();
            Scheduler?.Dispose();
            Unet?.Dispose();
            ImageProcessor?.Dispose();
            _ops?.Dispose();
        }

        internal struct Embeddings {
            public Tensor<float> PromptEmbeds { get; set; }
            public Tensor<float> NegativePromptEmbeds { get; set; }
            public Tensor<float> PooledPromptEmbeds { get; set; }
            public Tensor<float> NegativePooledPromptEmbeds { get; set; }
        }
    }
}
