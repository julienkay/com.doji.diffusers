using NUnit.Framework;
using Unity.Sentis;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class SentisUtilsTest {

        private float[] Samples {
            get {
                return TestUtils.LoadFromFile("256_latents");
            }
        }

        private float[] ExpectedQuantile {
            get {
                return TestUtils.LoadFromFile("quantile_test_result_256");
            }
        }

        private float[] ExpectedSorted{
            get {
                return TestUtils.LoadFromFile("sort_test_result_256");
            }
        }

        [Test]
        public void TestQuantile() {
            using Ops ops = WorkerFactory.CreateOps(BackendType.GPUCompute, null);
            using TensorFloat latents = new TensorFloat(new TensorShape(1, 4, 8, 8), Samples);
            TensorFloat quantile = ops.Quantile(latents, 0.995f, 1);
            quantile.MakeReadable();
            CollectionAssert.AreEqual(ExpectedQuantile, quantile.ToReadOnlyArray(), new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestSort() {
            using Ops ops = WorkerFactory.CreateOps(BackendType.GPUCompute, null);
            using TensorFloat latents = new TensorFloat(new TensorShape(1, 4, 8, 8), Samples);
            TensorFloat sorted = ops.Sort(latents, 1);
            sorted.MakeReadable();
            CollectionAssert.AreEqual(ExpectedSorted, sorted.ToReadOnlyArray(), new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestNonzero() {
            using Ops ops = WorkerFactory.CreateOps(BackendType.GPUCompute, null);
            float[] data = new float[] { 0, 1, 0, 2, 3, 0, 0, -1 };
            using TensorFloat test = new TensorFloat(new TensorShape(data.Length), data);
            TensorInt nonzero = ops.NonZero(test);
            nonzero.MakeReadable();
            CollectionAssert.AreEqual(new int[] { 1, 3, 4, 7 }, nonzero.ToReadOnlyArray());
        }
    }
}