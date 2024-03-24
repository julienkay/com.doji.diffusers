using NUnit.Framework;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.TestTools;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Test a <see cref="StableDiffusionXLPipeline"/>.
    /// Requires the models for stabilityai/stable-diffusion-xl-base-1.0 to be downloaded.
    /// </summary>
    public class PipelineTestSDXL : PipelineTestBase {

        [OneTimeSetUp]
        public void SetUp() {
            SetUp(DiffusionModel.SD_XL_BASE);
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