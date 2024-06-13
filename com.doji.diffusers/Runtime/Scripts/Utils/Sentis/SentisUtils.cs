using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Extends Ops class with not-yet implemented operators
    /// and some overloads for more convenience.
    /// </summary>
    public static class SentisUtils {

        private static Tensor[] _tmpTensorRefs = new Tensor[2];
        private static Tensor[] _tmpConcatTensorRefs = new Tensor[2];

        /// <summary>
        /// Computes the q-th quantiles of each row of the input tensor along the dimension dim.
        /// torch.quantile
        /// </summary>
        public static TensorFloat Quantile(this Ops ops, TensorFloat tensor, float q, int dim) {
            TensorFloat sorted = ops.Sort(tensor, dim);
            
            if (q < 0 || q > 1) {
                throw new ArgumentException("Quantile value must be between 0 and 1");
            }

            float index = (tensor.shape[dim] - 1) * q;

            using TensorInt lowerIndex = new TensorInt((int)MathF.Floor(index));
            using TensorInt upperIndex = new TensorInt((int)MathF.Ceiling(index));

            TensorFloat lowerValues = ops.Gather(sorted, lowerIndex, dim);
            TensorFloat upperValues = ops.Gather(sorted, upperIndex, dim);
            float weights = index - (int)MathF.Floor(index);

            TensorFloat sub = ops.Sub(upperValues, lowerValues);
            TensorFloat mul = ops.Mul(sub, weights);
            TensorFloat interpolated = ops.Add(mul, lowerValues);
            return interpolated;
        }

        public static TensorFloat Sort(this Ops ops, TensorFloat tensor, int dim) {
            int num = tensor.shape[dim];
            return ops.TopK(tensor, num, dim, largest: false /* sort lowest-to-highest */, true).values;
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
            BurstTensorData.Pin(X);
            int nbNonZeroIndices = 0;
            var end = X.shape.length;
            for (int i = 0; i < end; ++i) {
                if (X[i] != 0.0f)
                    nbNonZeroIndices += 1;
            }

            TensorInt nonZeroIndices = TensorInt.AllocZeros(new TensorShape(X.shape.rank, nbNonZeroIndices));
            if (nonZeroIndices.shape.HasZeroDims())
                return nonZeroIndices;

            BurstTensorData.Pin(nonZeroIndices, clearOnInit: false);
            int nonZeroIndicesIdx = 0;
            for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext()) {
                if (X[it.index] != 0.0f) {
                    for (int i = 0; i < X.shape.rank; i++)
                        nonZeroIndices[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                    nonZeroIndicesIdx++;
                }
            }

            return nonZeroIndices;
        }

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
            if (_tmpConcatTensorRefs.Length != tensors.Count) {
                _tmpConcatTensorRefs = new Tensor[tensors.Count];
            }
            for (int i = 0; i < tensors.Count; i++) {
                _tmpConcatTensorRefs[i] = tensors[i];
            }
            return ops.Concat(_tmpTensorRefs, axis) as TensorFloat;
        }

        public static TensorFloat Concatenate(this Ops ops, TensorFloat tensor1, TensorFloat tensor2, int axis = 0) {
            return ops.Concatenate(tensor1 as Tensor, tensor2 as Tensor, axis) as TensorFloat;
        }

        public static TensorInt Concatenate(this Ops ops, TensorInt tensor1, TensorInt tensor2, int axis = 0) {
            return ops.Concatenate(tensor1 as Tensor, tensor2 as Tensor, axis) as TensorInt;
        }

        /// <summary>
        /// Similar to torch.repeat() or numpy.tile()
        /// </summary>
        public static TensorFloat Repeat(this Ops ops, TensorFloat tensor, int repeats, int axis) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            if (repeats == 1) {
                return tensor;
            }

            int[] r = ArrayUtils.Full(tensor.shape.rank, 1);
            r[axis] = repeats;
            return ops.Tile(tensor, r);
        }

        /// <summary>
        /// Similar to torch.repeat() or numpy.tile()
        /// </summary>
        public static TensorInt Repeat(this Ops ops, TensorInt tensor, int repeats, int axis) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            if (repeats == 1) {
                return tensor;
            }

            int[] r = ArrayUtils.Full(tensor.shape.rank, 1);
            r[axis] = repeats;
            return ops.Tile(tensor, r);
        }


        /// <summary>
        /// Similar to torch.repeat_interleave() or numpy.repeat()
        /// </summary>
        public static TensorFloat RepeatInterleave(this Ops ops, TensorFloat tensor, int repeats, int dim) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            // implement repeat_interleave using repeat, reshape & transpose ops
            var repeat = ops.Repeat(tensor, repeats, dim);
            repeat.Reshape(repeat.shape.Flatten());
            var flatShape = repeat.shape;
            repeat.Reshape(new TensorShape(repeats, flatShape.length / repeats));
            var transpose = ops.Transpose(repeat, new int[] { 1, 0 });
            transpose.Reshape(flatShape);
            return transpose;
        }

        /// <summary>
        /// Similar to torch.repeat_interleave() or numpy.repeat()
        /// </summary>
        public static TensorInt RepeatInterleave(this Ops ops, TensorInt tensor, int repeats, int dim) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            // implement repeat_interleave using repeat, reshape & transpose ops
            var repeat = ops.Repeat(tensor, repeats, dim);
            repeat.Reshape(repeat.shape.Flatten());
            var flatShape = repeat.shape;
            repeat.Reshape(new TensorShape(repeats, flatShape.length / repeats));
            var transpose = ops.Transpose(repeat, new int[] { 1, 0 });
            transpose.Reshape(flatShape);
            return transpose;
        }

        /// <summary>
        /// Alias for <see cref="Ops.Split{T}(T, int, int, int)"/> to match the
        /// arguments of numpy.split() or torch.chunk() i.e. providing <paramref name="sections"/>
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