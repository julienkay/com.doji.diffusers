using System;
using Unity.Sentis;

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

        public static TensorFloat Clamp(this Ops ops, TensorFloat tensor, TensorFloat min, TensorFloat max) {
            return ops.Min(ops.Max(tensor, min), max);
        }

        /// <summary>
        /// Returns the indices of the elements of the input tensor that are not zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public static TensorInt NonZero(this Ops ops, TensorFloat X) {
            ArrayTensorData.Pin(X);
            int nbNonZeroIndices = 0;
            var end = X.shape.length;
            for (int i = 0; i < end; ++i) {
                if (X[i] != 0.0f)
                    nbNonZeroIndices += 1;
            }

            var tmpO = TensorInt.Zeros(new TensorShape(X.shape.rank, nbNonZeroIndices));
            if (tmpO.shape.HasZeroDims())
                return tmpO;

            ArrayTensorData.Pin(tmpO, clearOnInit: false);
            int nonZeroIndicesIdx = 0;
            for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext()) {
                if (X[it.index] != 0.0f) {
                    for (int i = 0; i < X.shape.rank; i++)
                        tmpO[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                    nonZeroIndicesIdx++;
                }
            }

            var O = ops.Copy(tmpO);
            tmpO.Dispose();
            return O;
        }
    }
}