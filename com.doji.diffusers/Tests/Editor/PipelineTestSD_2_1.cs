using NUnit.Framework;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Test a <see cref="StableDiffusionPipeline"/>.
    /// Requires the models for julienkay/stable-diffusion-2-1 to be downloaded.
    /// </summary>
    public class PipelineTestSD_2_1 : PipelineTestBase {

        [OneTimeSetUp]
        public void SetUp() {
            SetUp(DiffusionModel.SD_2_1);
        }

        [Test]
        public void TestSD21() {
            using var latents = LatentsLarge;
            Parameters p = new Parameters() {
                Prompt = "a cat",
                Width = 512,
                Height = 512,
                NumInferenceSteps = 20,
                GuidanceScale = 7.5f,
                Latents = latents,
            };

            var generated = _pipeline.Generate(p);

            TestUtils.ToFile(_pipeline, generated);
        }
    }
}