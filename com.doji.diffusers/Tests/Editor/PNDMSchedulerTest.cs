using NUnit.Framework;
using System.Collections;
using UnityEngine.TestTools.Utils;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Test the <see cref="TextEncoder"/> of a <see cref="StableDiffusionPipeline"/>.
    /// Requires the models for runwayml/stable-diffusion-v1-5 to be downloaded.
    /// </summary>
    public class PNDMSchedulerTest {

        private PNDMScheduler _scheduler;

        private float[] ExpectedBetas {
            get {
                return TestUtils.LoadFromFile("pndm_test_betas");
            }
        }

        /// <summary>
        /// Loads deterministic random samples with shape (4, 3, 8, 8)
        /// </summary>
        private float[] DummySamples {
            get {
                return TestUtils.LoadFromFile("pndm_test_random_samples");
            }
        }

        /// <summary>
        /// Load expected after running a full loop with deterministic inputs
        /// </summary>
        private float[] ExpectedOutput {
            get {
                return TestUtils.LoadFromFile("pndm_test_expected_output");
            }
        }

        [SetUp]
        public void SetUp() {
            _scheduler = new PNDMScheduler(
                  betaEnd: 0.012f,
                  betaSchedule: Schedule.ScaledLinear,
                  betaStart: 0.00085f,
                  numTrainTimesteps: 1000,
                  setAlphaToOne: false,
                  skipPrkSteps: true,
                  stepsOffset: 1,
                  trainedBetas: null
            );
        }

        [Test]
        public void TestInit() {
            Assert.That(_scheduler.Timesteps, Is.Not.Null);
            Assert.That(_scheduler.Timesteps.Length, Is.EqualTo(1000));
            for (int i = 0; i < 1000; i++) {
                Assert.That(_scheduler.Timesteps[i], Is.EqualTo(1000 - i - 1));
            }
        }

        /// <summary>
        /// Test the expected beta values for the default value <see cref="Schedule.ScaledLinear"/>
        /// after initialization
        /// </summary>
        [Test]
        public void TestBetas() {
            CollectionAssert.AreEqual(ExpectedBetas, _scheduler.Betas, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestFinalAlphaCumprod() {
            Assert.That(_scheduler.FinalAlphaCumprod, Is.EqualTo(0.9991f).Using(new FloatEqualityComparer(0.0001f)));
        }

        public static IEnumerable StepsTestData {
            get {
                yield return new TestCaseData(true).Returns(new int[] { 901, 801, 801, 701, 601, 501, 401, 301, 201, 101, 1 });
                yield return new TestCaseData(false).Returns(new int[] { 901, 851, 851, 801, 801, 751, 751, 701, 701, 651, 651, 601, 601, 501, 401, 301, 201, 101, 1 });
            }
        }

        [Test]
        [TestCaseSource(nameof(StepsTestData))]
        public int[] TestStepsOffset(bool skipPrkSteps) {
            var scheduler = new PNDMScheduler(
                  betaEnd: 0.02f,
                  betaSchedule: Schedule.Linear,
                  betaStart: 0.0001f,
                  numTrainTimesteps: 1000,
                  stepsOffset: 1,
                  skipPrkSteps: skipPrkSteps
            );
            scheduler.SetTimesteps(10);
            return scheduler.Timesteps;
        }

        [Test]
        public void TestFullLoop() {
            var scheduler = new PNDMScheduler(
                  betaEnd: 0.02f,
                  betaSchedule: Schedule.Linear,
                  betaStart: 0.0001f,
                  numTrainTimesteps: 1000,
                  stepsOffset: 1,
                  skipPrkSteps: true
            );
            scheduler.SetTimesteps(10);
            float[] sample = DummySamples;
           
            foreach(int t in scheduler.Timesteps) {
                var residual = Model(sample, t);
                sample = scheduler.Step(residual, t, sample).PrevSample;
                UnityEngine.Debug.Log(t);
                UnityEngine.Debug.Log(string.Join(", ", sample));
            }

            CollectionAssert.AreEqual(ExpectedOutput, sample, new FloatArrayComparer(0.00001f));
        }

        private float[] Model(float[] sample, int t) {
            float[] result = new float[sample.Length];
            for (int i = 0; i< sample.Length; i++) {
                result[i] = sample[i] * ((float)t / (t + 1));
            }
            return result;
        }
    }
}