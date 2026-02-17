using Unity.InferenceEngine;

namespace Doji.AI.Diffusers {

    /// <summary>
    /// Most utils are in <see cref="Doji.AI.SentisUtils"/>, here we only keep some of the
    /// more sketchy, half-baked extension methods, especially if they only work for certain shapes
    /// like flat tensors.
    /// </summary>
    public static class SentisUtils {

        /// <summary>
        /// Returns the indices of the elements of the input tensor that are not zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public static Tensor<int> NonZero(this Ops ops, Tensor<float> X) {
            throw new System.NotImplementedException();
            /*ComputeTensorData.Pin(X);
            int nbNonZeroIndices = 0;
            var end = X.shape.length;
            for (int i = 0; i < end; ++i) {
                if (X[i] != 0.0f)
                    nbNonZeroIndices += 1;
            }

            Tensor<int> nonZeroIndices = new Tensor<int>(new TensorShape(X.shape.rank, nbNonZeroIndices));
            if (nonZeroIndices.shape.HasZeroDims())
                return nonZeroIndices;

            ComputeTensorData.Pin(nonZeroIndices, clearOnInit: false);
            int nonZeroIndicesIdx = 0;
            for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext()) {
                if (X[it.index] != 0.0f) {
                    for (int i = 0; i < X.shape.rank; i++)
                        nonZeroIndices[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                    nonZeroIndicesIdx++;
                }
            }

            return nonZeroIndices;*/
        }
    }
}