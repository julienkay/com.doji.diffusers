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

        [Test]
        public void TestLinspaceWhole() {
            float[] spacedValues = ArrayUtils.Linspace(1, 15, 15);
            CollectionAssert.AreEqual(spacedValues, new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f});
        }


        [Test]
        public void TestLinspaceFract1() {
            float[] spacedValues = ArrayUtils.Linspace(1, 20, 15, endpoint: false);
            Assert.That(spacedValues,
                Is.EqualTo(
                    new float[] { 1.0f, 2.26666667f, 3.53333333f, 4.8f, 6.06666667f, 7.33333333f, 8.6f, 9.86666667f, 11.13333333f, 12.4f, 13.66666667f, 14.93333333f, 16.2f, 17.46666667f, 18.73333333f }
                ).Using(new FloatArrayComparer(0.000001f))
            );
        }

        [Test]
        public void TestLinspaceFract2() {
            float[] spacedValues = ArrayUtils.Linspace(1, 20, 15, endpoint: true);
            Assert.That(spacedValues,
              Is.EqualTo(
                  new float[] { 1.0f, 2.35714286f, 3.71428571f, 5.07142857f, 6.42857143f, 7.78571429f, 9.14285714f, 10.5f, 11.85714286f, 13.21428571f, 14.57142857f, 15.92857143f, 17.28571429f, 18.64285714f, 20.0f }
              ).Using(new FloatArrayComparer(0.000001f))
            );
        }
    }
}