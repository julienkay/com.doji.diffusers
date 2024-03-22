using Doji.AI.Transformers;
using Newtonsoft.Json;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Holds the parameters that are used to generate an image.
    /// </summary>
    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct Parameters {

        /* general txt2img parameters */

        /// <summary>
        /// The prompt or prompts to guide the image generation.
        /// </summary>
        [JsonProperty("prompt")]
        public Input Prompt { get; set; }

        [JsonIgnore]
        public readonly string PromptString => Prompt.ToString();

        /// <summary>
        /// The height in pixels of the generated image.
        /// </summary>
        [JsonProperty("height")]
        public int? Height { get; set; }

        /// <summary>
        /// The width in pixels of the generated image.
        /// </summary>
        [JsonProperty("width")]
        public int? Width { get; set; }

        /// <summary>
        /// The number of denoising steps.
        /// More denoising steps usually lead to a higher quality image at the expense of slower inference.
        /// 
        /// In the case of img2img pipelines this parameter will be modulated by <paramref name="strength"/>.
        /// </summary>
        [JsonProperty("num_inference_steps")]
        public int? NumInferenceSteps { get; set; }

        /// <summary>
        /// Guidance scale as defined in <see href="https://arxiv.org/abs/2207.12598">Classifier-Free Diffusion Guidance</see>.
        /// `guidance_scale` is defined as `w` of equation 2. of <see href="https://arxiv.org/pdf/2205.11487.pdf">Imagen Paper</see>.
        /// Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
        /// that are closely linked to the text `prompt`, usually at the expense of lower image quality.
        /// </summary>
        [JsonProperty("guidance_scale")]
        public float? GuidanceScale { get; set; }

        /// <summary>
        /// The prompt or prompts not to guide the image generation.
        /// Ignored when not using guidance (i.e., ignored if <see cref="GuidanceScale"/> is less than `1`).
        /// </summary>
        [JsonProperty("negative_prompt")]
        public Input NegativePrompt { get; set; }

        [JsonIgnore]
        public readonly string NegativePromptString => NegativePrompt.ToString();

        /// <summary>
        /// The number of images to generate per prompt.
        /// (HINT: Values > 1 not supported yet.)
        /// </summary>
        [JsonProperty("num_images_per_prompt")]
        public int? NumImagesPerPrompt { get; set; }

        /// <summary>
        /// Corresponds to parameter eta in the <see href="https://arxiv.org/abs/2010.02502">DDIM paper</see>.
        /// Only applies to <see cref="DDIMScheduler"/>, will be ignored for others.
        /// </summary>
        [JsonProperty("eta")]
        public float? Eta { get; set; }

        /// <summary>
        /// A seed to use to generate initial noise. Set this to make generation deterministic.
        /// </summary>
        [JsonProperty("seed")]
        public uint? Seed { get; set; }

        /// <summary>
        /// Pre-generated noise, sampled from a Gaussian distribution, to be used as inputs for image generation.
        /// If not provided, a latents tensor will be generated for you using the supplied <see cref="Seed"/>.
        /// </summary>
        [JsonIgnore]
        public TensorFloat Latents { get; set; }

        /// <summary>
        /// Guidance rescale factor proposed by <see href="https://arxiv.org/pdf/2305.08891.pdf">
        /// Common Diffusion Noise Schedules and Sample Steps are Flawed</see>.
        /// <see cref="GuidanceScale"/> is defined as `phi` in equation 16.
        /// Guidance rescale factor should fix overexposure when using zero terminal SNR.
        /// </summary>
        [JsonProperty("guidance_rescale")]
        public float? GuidanceRescale { get; set; }

        /// <summary>
        /// A function that will be called at every step during inference.
        /// </summary>
        [JsonIgnore]
        public PipelineCallback Callback { get; set; }

        /* img2img parameters */

        /// <summary>
        /// Tensor representing an image, that will be used as the starting point for the process.
        /// </summary>
        [JsonIgnore]
        public TensorFloat Image { get; set; }

        /// <summary>
        /// Conceptually, indicates how much to transform the reference <see cref="Image"/>. Must be between 0 and 1.
        /// The image will be used as a starting point, adding more noise to it the larger the <see cref="Strength"/>.
        /// number of denoising steps depends on the amount of noise initially added.
        /// When <paramref name="strength"/> is 1, added noise will be maximum and the denoising process will run for
        /// the full number of iterations specified in <see cref="NumInferenceSteps"/>.
        /// A value of 1, therefore, essentially ignores <see cref="Image"/>.
        /// </summary>
        [JsonProperty("strength")]
        public float? Strength { get; set; }

        /* sdxl parameters */

        /// <summary>
        /// If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
        /// `original_size` defaults to `(<see cref="Height"/>, <see cref="Width"/>)` if not specified. Part of
        /// SDXL's micro-conditioning as explained in section 2.2 of <see href="https://huggingface.co/papers/2307.01952">
        /// SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis</see>.
        /// </summary>
        [JsonProperty("original_size")]
        public (int width, int height)? OriginalSize { get; set; }

        /// <summary>
        /// `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
        /// `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
        /// `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
        /// <see href="https://huggingface.co/papers/2307.01952">SDXL: Improving Latent Diffusion Models for
        /// High-Resolution Image Synthesis</see>.
        /// </summary>
        [JsonProperty("crop_coords_top_left")]
        public (int x, int y)? CropsCoordsTopLeft { get; set; }

        /// <summary>
        /// For most cases, <see cref="TargetSize>"/> should be set to the desired height and width of the generated image.
        /// If not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
        /// section 2.2 of <see href="https://huggingface.co/papers/2307.01952">SDXL: Improving Latent Diffusion Models
        /// for High-Resolution Image Synthesis</see>.
        /// </summary>
        [JsonProperty("target_size")]
        public (int width, int height)? TargetSize { get; set; }

        /* sdxl img2img parameters */

        /// <summary>
        /// Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
        /// Part of SDXL's micro-conditioning as explained in section 2.2 of <see href="https://huggingface.co/papers/2307.01952">
        /// SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis</see>.
        /// </summary>
        [JsonProperty("aesthetic_score")]
        public float? AestheticScore { get; set; }

        /// <summary>
        /// Part of SDXL's micro-conditioning as explained in section 2.2 of<see href="https://huggingface.co/papers/2307.01952">
        /// SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis</see>. Can be used to simulate
        /// an aesthetic score of the generated image by influencing the negative text condition.
        /// </summary>
        [JsonProperty("negative_aesthetic_score")]
        public float? NegativeAestheticScore { get; set; }
    }
}