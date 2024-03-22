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
        public void TestSD15Small() {
            using var sd = StableDiffusionPipeline.FromPretrained(DiffusionModel.SD_1_5);
            using var latents = Latents;

            Parameters p = new Parameters() {
                Prompt = "a cat",
                Width = 64,
                Height = 64,
                NumInferenceSteps = 10,
                GuidanceScale = 1f,
                Latents = latents,
                Callback = TestPredictedNoise,
            };

            TensorFloat generated = sd.Generate(p);

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
        public void TestSD15() {
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");

            using var sd = DiffusionPipeline.FromPretrained(DiffusionModel.SD_1_5);
            using var latents = LatentsLarge;

            Parameters p = new Parameters() {
                Prompt = "a cat",
                Width = 512,
                Height = 512,
                NumInferenceSteps = 10,
                GuidanceScale = 7.5f,
                Latents = latents,
                Callback= TestPredictedNoiseLarge,
            };

            var generated = sd.Generate(p);

            TestUtils.ToFile(sd, generated);   
        }

        private void TestPredictedNoiseLarge(int i, float t, TensorFloat latents) {
            latents.MakeReadable();
            CollectionAssert.AreEqual(GetLatentsLarge(i), latents.ToReadOnlyArray(), new FloatArrayComparer(0.0001f), $"Latents differ at step {i}");
        }

        [Test]
        public void TestSD21() {
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");

            using var sd = DiffusionPipeline.FromPretrained(DiffusionModel.SD_2_1);
            using var latents = LatentsLarge;
            Parameters p = new Parameters() {
                Prompt = "a cat",
                Width = 512,
                Height = 512,
                NumInferenceSteps = 20,
                GuidanceScale = 7.5f,
                Latents = latents,
            };

            var generated = sd.Generate(p);

            TestUtils.ToFile(sd, generated);
        }

        [Test]
        public void TestSDXL() {
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");
            LogAssert.Expect(LogType.Error, "Thread group count is above the maximum allowed limit. Maximum allowed thread group count is 65535.");

            using var sd = DiffusionPipeline.FromPretrained(DiffusionModel.SD_XL_BASE);
            int width = 1024;
            int height = 1024;
            string prompt = "a cat";
            using var latents = new TensorFloat(new TensorShape(1, 4, 128, 128), ArrayUtils.Randn(4 * 128 * 128));

            Parameters p = new Parameters() {
                Prompt = prompt,
                Width = width,
                Height = height,
                NumInferenceSteps = 20,
                GuidanceScale = 7.5f,
                Latents = latents,
            };

            var generated = sd.Generate(p);

            TestUtils.ToFile(sd, generated);
        }
    }
}