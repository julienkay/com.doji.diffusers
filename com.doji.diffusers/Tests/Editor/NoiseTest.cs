using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class NoiseTest {
        private float[] Samples {
            get {
                return TestUtils.LoadFromFile("256_latents");
            }
        }

        /// Loads deterministic random samples with shape (1, 4, 64, 64)
        /// </summary>
        private float[] SamplesLarge {
            get {
                return TestUtils.LoadFromFile("16384_latents");
            }
        }

        [Test]
        public void TestSentisRandomNoise() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            var latents = ops.RandomNormal(new TensorShape(1, 4, 64, 64), 0, 1, new System.Random().Next());
            ops.ExecuteCommandBufferAndClear();
            float[] array = latents.DownloadToArray();
            Assert.That(Math.Abs(array.Average() - 0), Is.LessThan(0.05));
            Assert.That(Math.Abs(array.Variance() - 1), Is.LessThan(0.05));
        }

        [Test]
        public void TestRandomNoise() {
            var array = ArrayUtils.Randn(4096);

            // Mean close to 0
            Assert.That(Math.Abs(array.Average() - 0), Is.LessThan(0.05));

            // Variance close to 1
            Assert.That(Math.Abs(array.Variance() - 1), Is.LessThan(0.05));
        }

        [Test]
        public void TestPregeneratedNoise() {
            var array = Samples;

            // Mean close to 0
            Assert.That(Math.Abs(array.Average() - 0), Is.LessThan(0.05));

            // Variance close to 1
            Assert.That(Math.Abs(array.Variance() - 1), Is.LessThan(0.05));
        }

        [Test]
        public void TestPregeneratedNoiseLarge() {
            var array = SamplesLarge;

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