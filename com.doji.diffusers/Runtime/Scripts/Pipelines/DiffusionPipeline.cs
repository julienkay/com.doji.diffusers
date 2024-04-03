using Doji.AI.Transformers;
using System;
using System.Threading.Tasks;
using Unity.Sentis;

namespace Doji.AI.Diffusers {


    public delegate void PipelineCallback(int step, float timestep, TensorFloat latents);

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

        /* Parameters accessors for convenience */

        protected Input prompt { get => _parameters.Prompt; set => _parameters.Prompt = value; }
        protected int height { get => _parameters.Height.Value; set => _parameters.Height = value; }
        protected int width { get => _parameters.Width.Value; set => _parameters.Width = value; }
        protected int numInferenceSteps { get => _parameters.NumInferenceSteps.Value; set => _parameters.NumInferenceSteps = value; }
        protected float guidanceScale { get => _parameters.GuidanceScale.Value; set => _parameters.GuidanceScale = value; }
        protected Input negativePrompt { get => _parameters.NegativePrompt; set => _parameters.NegativePrompt = value; }
        protected int numImagesPerPrompt { get => _parameters.NumImagesPerPrompt.Value; set => _parameters.NumImagesPerPrompt = value; }
        protected float eta { get => _parameters.Eta.Value; set => _parameters.Eta = value; }
        protected uint? seed { get => _parameters.Seed; set => _parameters.Seed = value; }
        protected TensorFloat latents { get => _parameters.Latents; set => _parameters.Latents = value; }
        protected float guidanceRescale { get => _parameters.GuidanceRescale.Value; set => _parameters.GuidanceRescale = value; }
        protected PipelineCallback callback { get => _parameters.Callback; set => _parameters.Callback = value; }
        protected TensorFloat image { get => _parameters.Image; set => _parameters.Image = value; }
        protected float strength { get => _parameters.Strength.Value; set => _parameters.Strength = value; }
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
        public DiffusionPipeline(BackendType backendType) {
            // TODO: When casting between pipeline types, we might want to reuse ops and image processor as well
            _ops = WorkerFactory.CreateOps(backendType, null);
            if (VaeDecoder.Config.BlockOutChannels != null) {
                VaeScaleFactor = 1 << (VaeDecoder.Config.BlockOutChannels.Length - 1);
            } else {
                VaeScaleFactor = 8;
            }
            ImageProcessor = new VaeImageProcessor(vaeScaleFactor: VaeScaleFactor, backend: backendType);
        }

        /// <summary>
        /// Applies default values for the specific pipeline.
        /// </summary>
        /// <param name="parameters">The parameters that were passed to the Generate() method.</param>
        protected void SetParameterDefaults(Parameters parameters) {
            Parameters defaults = GetDefaultParameters();
            _parameters = parameters;
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
            if (this is IImg2ImgPipeline && image == null) {
                throw new ArgumentException($"Please provide an 'image' parameter to generate images.");
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

        /// <summary>
        /// Execute the pipeline to generate images.
        /// </summary>
        /// <param name="parameters">the parameters used to generate the image</param>
        /// <returns>the resulting image tensor</returns>
        public abstract TensorFloat Generate(Parameters parameters);

        /// <summary>
        /// Execute the pipeline to generate images.
        /// </summary>
        /// <remarks>
        /// This is an overload for the most common generation parameters for convenience.
        /// For more control and advanced pipeline usage, pass parameters via the
        /// <see cref="Generate(Parameters)"/> method instead.
        /// </remarks>
        public TensorFloat Generate(
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
        public Task<TensorFloat> GenerateAsync(
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
        public virtual Task<TensorFloat> GenerateAsync(Parameters parameters) {
            throw new NotImplementedException($"{nameof(GenerateAsync)} not implemented for {GetType().Name}.");
        }

        public virtual void Dispose() {
            VaeDecoder?.Dispose();
            TextEncoder?.Dispose();
            Scheduler?.Dispose();
            Unet?.Dispose();
            ImageProcessor?.Dispose();
            _ops?.Dispose();
        }
    }
}
