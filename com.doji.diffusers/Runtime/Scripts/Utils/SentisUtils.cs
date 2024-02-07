using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public static class SentisUtils {

        public static TensorFloat Quantile(this Ops ops, TensorFloat tensor, float q, int dim) {
            TensorFloat sorted = ops.Sort(tensor, dim);
            
            if (q < 0 || q > 1) {
                throw new ArgumentException("Quantile value must be between 0 and 1");
            }

            float index = (sorted.shape[dim] - 1) * q;

            using TensorInt lowerIndex = new TensorInt((int)MathF.Floor(index));
            using TensorInt upperIndex = new TensorInt((int)MathF.Ceiling(index));

            TensorFloat lowerValues = ops.Gather(sorted, lowerIndex, dim);
            TensorFloat upperValues = ops.Gather(sorted, upperIndex, dim);
            float weights = index - (int)MathF.Floor(index);

            TensorFloat tmp = ops.Sub(upperValues, lowerValues);
            TensorFloat tmp2 = ops.Mul(weights, tmp);
            TensorFloat interpolated = ops.Add(tmp2, lowerValues);

            return interpolated;
        }

        public static TensorFloat Sort(this Ops ops, TensorFloat tensor, int dim) {
            int num = tensor.shape[dim];
            return ops.TopK(tensor, num, dim, false, true)[0] as TensorFloat;
        }
    }
}