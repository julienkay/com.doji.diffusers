using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class ArrayUtilsTest {

        [Test]
        public void TestRandomNoise() {
            var array = ArrayUtils.Randn(4096);

            // Mean close to 0
            Assert.That(Math.Abs(array.Average() - 0), Is.LessThan(0.05));

            // Variance close to 1
            Assert.That(Math.Abs(array.Variance() - 1), Is.LessThan(0.05));
        }
    }

    public static class Extensions {
        public static double Variance(this IEnumerable<float> values) {
            double mean = values.Average();
            double variance = values.Select(val => Math.Pow(val - mean, 2)).Average();
            return variance;
        }
    }
}