using NUnit.Framework;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class ArrayUtilsTest {

        [Test]
        public void TestInterpolate() {
            float[] xp = { 1, 2, 3, 4, 5 };
            float[] fp = { 10, 20, 30, 40, 50 };
            float[] x = { 2.5f, 3.5f, 4.5f };

            float[] interpolatedValues = ArrayUtils.Interpolate(x, xp, fp, 0f, 0f);

            CollectionAssert.AreEqual(interpolatedValues, new float[] { 25.0f, 35.0f, 45.0f });
        }
    }
}