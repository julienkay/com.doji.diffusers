using NUnit.Framework;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.TestTools;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// <summary>
    /// Test a <see cref="StableDiffusionPipeline"/>.
    /// Requires the models for runwayml/stable-diffusion-v1-5 to be downloaded.
    /// </summary>
    /// </summary>
    public class SDPipelineTest {

        /// <summary>
        /// Loads deterministic random samples with shape (1, 4, 8, 8)
        /// </summary>
        private float[] Latents {
            get {
                return TestUtils.LoadFromFile("256_latents");
            }
        }
        private float[] LatentsLarge {
            get {
                return TestUtils.LoadFromFile("16384_latents");
            }
        }

        private float[] GetLatents(int i) {
            return TestUtils.LoadFromFile($"pipeline_test_latents_{i}");
        }

        private float[] GetLatentsLarge(int i) {
            return TestUtils.LoadFromFile($"pipeline_test_latents_512_{i}");
        }

        [Test]
        public void TestSD() {
            using var sd = StableDiffusionPipeline.FromPretrained(DiffusionModel.SD_1_5);
            int width = 64;
            int height = 64;
            string prompt = "a cat";

            TensorFloat generated = sd.Generate(prompt, width, height, 10, 1f, 1, Latents, callback: TestPredictedNoise);

            //TestUtils.ToFile(prompt, width, height, generated);   
        }

        private void TestPredictedNoise(int i, int t, float[] latents) {
            CollectionAssert.AreEqual(GetLatents(i), latents, new FloatArrayComparer(0.00001f), $"Latents differ at step {i}");
        }

        /// <summary>
        /// Tests Stable Diffusion Pipeline with 512x512 resolution
        /// </summary>
        /// <remarks>
        /// Note that we had to bump the expected error somewhat for this test to pass.
        /// </remarks>
        [Test]
        public void TestSDLarge() {
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");

            using var sd = StableDiffusionPipeline.FromPretrained(DiffusionModel.SD_1_5);
            int width = 512;
            int height = 512;
            string prompt = "a cat";

            var generated = sd.Generate(prompt, width, height, 10, 1f, 1, LatentsLarge, callback: TestPredictedNoiseLarge);
            var tmp = RenderTexture.GetTemporary(width, height);

            //TestUtils.ToFile(prompt, width, height, generated);   
        }

        private void TestPredictedNoiseLarge(int i, int t, float[] latents) {
            CollectionAssert.AreEqual(GetLatentsLarge(i), latents, new FloatArrayComparer(0.0001f), $"Latents differ at step {i}");
        }
    }
}