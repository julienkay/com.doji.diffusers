using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public enum Resample {
        Nearest,
        Bilinear,
        Bicubic,
        Lanczos
    }

    /// <summary>
    /// Image processor for VAE.
    /// </summary>
    public class VaeImageProcessor : IDisposable {

        private Ops _ops;

        private bool _doNormalize;

        /// <param name="doResize">
        /// Whether to downscale the image's (height, width) dimensions to multiples of <paramref name="vaeScaleFactor"/>.
        /// Can accept `height` and `width` arguments from <see cref="Preprocess"/> method.</param>
        /// <param name="vaeScaleFactor"> VAE scale factor. If `do_resize` is `True`, the image is automatically resized
        /// to multiples of this factor.</param>
        /// <param name="resample">Resampling filter to use when resizing the image.</param>
        /// <param name="doNormalize">Whether to normalize the image to [-1,1].</param>
        /// <param name="doBinarize">Whether to binarize the image to 0/1.</param>
        /// <param name="doConvertRgb">Whether to convert the images to RGB format.</param>
        /// <param name="doConvertGrayscale">Whether to convert the images to grayscale format.</param>
        public VaeImageProcessor(
            bool doResize = true,
            int vaeScaleFactor = 8,
            Resample resample = Resample.Lanczos,
            bool doNormalize = true,
            bool doBinarize = false,
            bool doConvertRgb = false,
            bool doConvertGrayscale = false,
            BackendType backend = BackendType.GPUCompute)
        {
            _doNormalize = doNormalize;
            if (doConvertRgb && doConvertGrayscale) {
                doConvertRgb = false;
                throw new ArgumentException("`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`," +
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`." +
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`");
            }
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        public TensorFloat PreProcess(
            TensorFloat image,
            int height = -1,
            int width = -1)
        {
            if (image.shape != new TensorShape(1, 3, 512, 512)) {
                throw new NotImplementedException("Image resizing not implemented yet. Make sure to pass a 512x512 RGB texture as an input image.");
            }

            // expected range [0,1], normalize to [-1,1]
            if (_doNormalize) {
                Normalize(image);
            }

            return image;
        }

        public TensorFloat PostProcess(TensorFloat image, bool? doDenormalize = null) {
            if (doDenormalize.HasValue || _doNormalize) {
                image = Denormalize(image);
            }
            return image;
        }

        /// <summary>
        /// Normalize an image tensor from [0,1] to [-1,1].
        /// </summary>
        private TensorFloat Normalize(TensorFloat image) {
            return _ops.Mad(image, 2.0f, -1.0f);
        }

        /// <summary>
        /// Denormalize an image tensor from [-1,1] to [0,1].
        /// </summary>
        private TensorFloat Denormalize(TensorFloat image) {
            return _ops.Mad(image, 0.5f, 0.5f);
        }

        public void Dispose() {
            _ops?.Dispose();
        }
    }
}