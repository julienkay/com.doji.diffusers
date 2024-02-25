using System;
using System.Buffers;
using System.Collections.Generic;
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

        private static readonly Tensor[] _tmpTensorRefs = new Tensor[2];

        /// <summary>
        /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> to match numpy.concatenate()
        /// naming and for convenience of not needing to create a Tensor array.
        /// </summary>
        public static Tensor Concatenate(this Ops ops, Tensor tensor1, Tensor tensor2, int axis = 0) {
            _tmpTensorRefs[0] = tensor1;
            _tmpTensorRefs[1] = tensor2;
            return ops.Concat(_tmpTensorRefs, axis);
        }

        /// <summary>
        /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> to match numpy.concatenate()
        /// naming and for convenience by adding a List<TensorFloat> overload.
        /// </summary>
        public static TensorFloat Concatenate(this Ops ops, List<TensorFloat> tensors, int axis = 0) {
            TensorFloat[] tensorArray = ArrayPool<TensorFloat>.Shared.Rent(tensors.Count);
            var result = ops.Concat(tensorArray, axis);
            ArrayPool<TensorFloat>.Shared.Return(tensorArray);
            return result as TensorFloat;
        }

        public static TensorFloat Concatenate(this Ops ops, TensorFloat tensor1, TensorFloat tensor2, int axis = 0) {
            return ops.Concatenate(tensor1 as Tensor, tensor2 as Tensor, axis) as TensorFloat;
        }

        /// <summary>
        /// numpy.repeat()
        /// </summary>
        /// <remarks>
        /// TODO: Implement this using <see cref="Ops.Tile{T}(T, ReadOnlySpan{int})"/>
        /// to support multiple images per prompt
        /// </remarks>
        public static TensorFloat Repeat(this Ops ops, TensorFloat tensor, int repeats, int axis) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }
            
            if (repeats == 1) {
                return tensor;
            }

            throw new NotImplementedException();
        }

        /// <summary>
        /// Alias for <see cref="Ops.Split{T}(T, int, int, int)"/> to match the
        /// arguments of numpy.split() i.e. providing <paramref name="sections"/>
        /// that the original tensor is split into.
        /// </summary>
        public static void Split(this Ops ops, Tensor tensor, int sections, int axis = 0, List<TensorFloat> splitTensors = null) {
            if (tensor.shape[axis] % sections != 0) {
                throw new ArgumentException($"Tensor dimension {axis} (length: {tensor.shape[axis]}) can not be divided into {sections} sections.");
            }
            splitTensors ??= new List<TensorFloat>();
            splitTensors.Clear();

            int step = tensor.shape[axis] / sections;
            int end = tensor.shape[axis] - step;
            for (int i = 0; i < end; i += step) {
                var section = ops.Split(tensor, axis: axis, i, i + step) as TensorFloat;
                splitTensors.Add(section);
            }
        }

        /// <summary>
        /// Splits a tensor into two sections.
        /// </summary>
        public static (TensorFloat a, TensorFloat b) SplitHalf(this Ops ops, Tensor tensor, int axis = 0) {
            if (tensor.shape[axis] % 2 != 0) {
                throw new ArgumentException($"Tensor dimension {axis} (length: {tensor.shape[axis]}) can not be divided into 2 sections.");
            }
            int half = tensor.shape[axis] / 2;
            int start = 0;
            int end = tensor.shape[axis];    
            var a = ops.Split(tensor, axis: axis, start, half) as TensorFloat;
            var b = ops.Split(tensor, axis: axis, half, end) as TensorFloat;
            return (a, b);
        }
    }
}