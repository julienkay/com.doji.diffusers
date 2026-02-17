using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public class MarigoldImageProcessor {

        private float _vaeScaleFactor;
        private bool _doNormalize;
        private bool _doRangeCheck;
        private Ops _ops;

        public MarigoldImageProcessor(
            float vaeScaleFactor = 8,
            bool doNormalize = true,
            bool doRangeCheck = true,
             BackendType backend = BackendType.GPUCompute)
        {
            _vaeScaleFactor = vaeScaleFactor;
            _doNormalize = doNormalize;
            _doRangeCheck = doRangeCheck;
            _ops = new Ops(backend);
        }

        public (Tensor<float>, Vector2Int, Vector2Int) Preprocess(
            Tensor<float> image,
            int? processingResolution,
            ResampleMethod resampleMethodInput = ResampleMethod.Bilinear)
        {
            int originalHeight = image.shape[2];
            int originalWidth = image.shape[3];
            Vector2Int originalResolution = new Vector2Int(originalHeight, originalWidth);

            if (_doRangeCheck) {
                CheckImageValuesRange(image);
            }

            if (_doNormalize) {
                _ops.Mad(image, 2.0f, -1.0f);
            }

            if (processingResolution != null && processingResolution.Value > 0) {
                image = ResizeToMaxEdge(image, processingResolution.Value, resampleMethodInput); // [N,3,PH,PW]
            }

            (var i, var padding) = PadImage(image, _vaeScaleFactor); // [N,3,PPH,PPW]
            image = i;

            return (image, padding, originalResolution);
        }

        private Tensor<float> ResizeAntialias(Tensor<float> image, int newW, int newH, ResampleMethod mode, bool? isAA = null) {
            if (image.shape.rank != 4) {
                throw new ArgumentException($"Invalid input dimensions; shape={image.shape}.");
            }
            //bool antialias = is_aa == true && (mode == ResampleMethod.Bilinear || mode == ResampleMethod.Bicubic);
            ReadOnlySpan<float> scale = new float[] { 1f, 1f, (float)newW / image.shape[2], (float)newH / image.shape[3] };
            image = _ops.Resize(image, scale, GetInterpolationMode(mode));

            return image;
        }

        private InterpolationMode GetInterpolationMode(ResampleMethod mode) {
            switch (mode) {
                case ResampleMethod.Nearest:
                    return InterpolationMode.Nearest;
                case ResampleMethod.Bilinear:
                    return InterpolationMode.Linear;
                case ResampleMethod.Bicubic:
                    return InterpolationMode.Cubic;
                case ResampleMethod.NearestExact:
                case ResampleMethod.Area:
                default:
                    throw new ArgumentException($"Resample strategy {mode} not supported.");
            }
        }

        private Tensor<float> ResizeToMaxEdge(Tensor<float> image, int maxEdgeSz, ResampleMethod mode) {
            if (image.shape.rank != 4) {
                throw new ArgumentException($"Invalid input dimensions; shape={image.shape}.");
            }
            int h = image.shape[-2];
            int w = image.shape[-1];
            int max_orig = Math.Max(h, w);
            int newH = h * maxEdgeSz; // max_orig
            int newW = w * maxEdgeSz; // max_orig

            if (newH == 0 || newW == 0) {
                throw new ArgumentException($"Extreme aspect ratio of the input image: [{w} x {h}]");
            }

            image = ResizeAntialias(image, newH, newW, mode, isAA: true);

            return image;
        }


        private (Tensor<float>, Vector2Int) PadImage(Tensor<float> image, float vaeScaleFactor) {
            throw new NotImplementedException();
        }

        private void CheckImageValuesRange(Tensor<float> image) {
           // skip this for now
        }
    }
} 