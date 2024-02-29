using NUnit.Framework;
using System.Collections;
using Unity.Sentis;
using UnityEngine.TestTools.Utils;
using static Doji.AI.Diffusers.Scheduler;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Tests for <see cref="PNDMScheduler"/>.
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
        private TensorFloat DummySamples {
            get {
                return TestUtils.LoadTensorFromFile("scheduler_test_random_samples", new TensorShape(4, 3, 8, 8));
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

        private StepArgs _stepArgs = new StepArgs();
        private Ops _ops;

        [SetUp]
        public void SetUp() {
            var config = new SchedulerConfig() {
                BetaEnd = 0.012f,
                BetaSchedule = Schedule.ScaledLinear,
                BetaStart = 0.00085f,
                NumTrainTimesteps = 1000,
                SetAlphaToOne = false,
                SkipPrkSteps = true,
                StepsOffset = 1,
                TrainedBetas = null
            };
            _scheduler = new PNDMScheduler(config);
            _ops = WorkerFactory.CreateOps(BackendType.GPUCompute, null);
        }

        [TearDown]
        public void TearDown() {
            _scheduler?.Dispose();
            _ops?.Dispose();
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
            _scheduler.Betas.MakeReadable();
            var betas = _scheduler.Betas.ToReadOnlyArray();
            CollectionAssert.AreEqual(ExpectedBetas, betas, new FloatArrayComparer(0.00001f));
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
            var config = new SchedulerConfig() {
                BetaEnd = 0.02f,
                BetaSchedule = Schedule.Linear,
                BetaStart = 0.0001f,
                NumTrainTimesteps = 1000,
                StepsOffset = 1,
                SkipPrkSteps = skipPrkSteps,
            };
            using var scheduler = new PNDMScheduler(config);
            scheduler.SetTimesteps(10);
            return scheduler.Timesteps;
        }

        [Test]
        public void TestFullLoop() {
            var config = new SchedulerConfig() {
                BetaEnd = 0.02f,
                BetaSchedule = Schedule.Linear,
                BetaStart = 0.0001f,
                NumTrainTimesteps = 1000,
                StepsOffset = 1,
                SkipPrkSteps = true,
            };
            using var scheduler = new PNDMScheduler(config);
            scheduler.SetTimesteps(10);
            var sample = DummySamples;
           
            foreach(int t in scheduler.Timesteps) {
                var residual = Model(sample, t);
                _stepArgs.Set(residual, t, sample);
                sample = scheduler.Step(_stepArgs).PrevSample;
            }

            sample.MakeReadable();
            CollectionAssert.AreEqual(ExpectedOutput, sample.ToReadOnlyArray(), new FloatArrayComparer(0.00001f));
        }

        private TensorFloat Model(TensorFloat sampleTensor, int t) {
            return _ops.Mul(sampleTensor, (float)t / (t + 1));
        }
    }
}