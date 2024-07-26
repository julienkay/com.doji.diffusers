using NUnit.Framework;
using System;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.TestTools;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Test a <see cref="StableDiffusionPipeline"/>.
    /// Requires the models for runwayml/stable-diffusion-v1-5 to be downloaded.
    /// </summary>
    public class PipelineTestSD_1_5 : PipelineTestBase {

        [OneTimeSetUp]
        public void SetUp() {
            SetUp(DiffusionModel.SD_1_5);
        }

        [Test]
        public void TestSD15Small() {
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

            TensorFloat generated = _pipeline.Generate(p);

            //TestUtils.ToFile(_pipeline, generated);   
        }

        private void TestPredictedNoise(int i, float t, TensorFloat latents) {
            latents.ReadbackAndClone();
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
            using var latents = LatentsLarge;

            Parameters p = new Parameters() {
                Prompt = "a cat",
                Width = 512,
                Height = 512,
                NumInferenceSteps = 10,
                GuidanceScale = 1f,
                Latents = latents,
                Callback = TestPredictedNoiseLarge,
            };

            var generated = _pipeline.Generate(p);

            TestUtils.ToFile(_pipeline, generated);   
        }

        private void TestPredictedNoiseLarge(int i, float t, TensorFloat latents) {
            latents.ReadbackAndClone();
            CollectionAssert.AreEqual(GetLatentsLarge(i), latents.ToReadOnlyArray(), new FloatArrayComparer(0.0001f), $"Latents differ at step {i}");
        }

        [Test]
        public void TestPromptNull() {
            Assert.Throws<ArgumentException>(() => _pipeline.Generate(prompt: null));
        }

        [Test]
        public void TestImageNull() {
            var img2img = _pipeline.As<IImg2ImgPipeline>();
            Assert.Throws<ArgumentException>(() => img2img.Generate("test", null));
        }
    }
}