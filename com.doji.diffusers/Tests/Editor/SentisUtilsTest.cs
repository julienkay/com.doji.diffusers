using NUnit.Framework;
using Unity.Sentis;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class SentisUtilsTest {

        [Test]
        public void TestNonzero() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            float[] data = new float[] { 0, 1, 0, 2, 3, 0, 0, -1 };
            using TensorFloat test = new TensorFloat(new TensorShape(data.Length), data);
            TensorInt nonzero = ops.NonZero(test);
            nonzero.ReadbackAndClone();
            CollectionAssert.AreEqual(new int[] { 1, 3, 4, 7 }, nonzero.ToReadOnlyArray());
        }
    }
}