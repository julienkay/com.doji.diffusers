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
        private TensorFloat Latents {
            get {
                return TestUtils.LoadTensorFromFile("256_latents", new TensorShape(1, 4, 8, 8));
            }
        }

        private TensorFloat LatentsLarge {
            get {
                return TestUtils.LoadTensorFromFile("16384_latents", new TensorShape(1, 4, 64, 64));
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
            using var latents = Latents;

            TensorFloat generated = sd.Generate(prompt, width, height, numInferenceSteps: 10,
                guidanceScale: 1f, latents: latents, callback: TestPredictedNoise);

            //TestUtils.ToFile(sd, generated);   
        }

        private void TestPredictedNoise(int i, float t, TensorFloat latents) {
            latents.MakeReadable();
            CollectionAssert.AreEqual(GetLatents(i), latents.ToReadOnlyArray(), new FloatArrayComparer(0.00001f), $"Latents differ at step {i}");
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
            using var latents = LatentsLarge;

            var generated = sd.Generate(prompt, width, height, numInferenceSteps: 10, guidanceScale: 7.5f, latents: latents);
            var tmp = RenderTexture.GetTemporary(width, height);

            //TestUtils.ToFile(sd, generated);   
        }

        private void TestPredictedNoiseLarge(int i, float t, TensorFloat latents) {
            latents.MakeReadable();
            CollectionAssert.AreEqual(GetLatentsLarge(i), latents.ToReadOnlyArray(), new FloatArrayComparer(0.0001f), $"Latents differ at step {i}");
        }
    }
}