using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine.TestTools.Utils;
using static Doji.AI.Diffusers.Scheduler;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Tests for <see cref="EulerDiscreteScheduler"/>.
    /// </summary>
    public class EulerDiscreteSchedulerTest {

        private EulerDiscreteScheduler _scheduler;

        private float[] ExpectedBetas {
            get {
                return TestUtils.LoadFromFile("euler_discrete_test_betas");
            }
        }

        /// <summary>
        /// Loads deterministic random samples with shape (4, 3, 8, 8)
        /// </summary>
        private Tensor<float> DummySamples {
            get {
                return TestUtils.LoadTensorFromFile("scheduler_test_random_samples", new TensorShape(4, 3, 8, 8));
            }
        }

        /// <summary>
        /// Load expected after running a full loop with deterministic inputs
        /// </summary>
        private float[] ExpectedOutput {
            get {
                return TestUtils.LoadFromFile("euler_discrete_test_expected_output");
            }
        }

        private Ops _ops;

        [SetUp]
        public void SetUp() {
            var config = new SchedulerConfig() {
                NumTrainTimesteps = 1000,
                BetaEnd = 0.012f,
                BetaSchedule = Schedule.ScaledLinear,
                BetaStart = 0.00085f,
                InterpolationType = Interpolation.Linear,
                PredictionType = Prediction.Epsilon,
                StepsOffset = 1,
                TimestepSpacing = Spacing.Leading,
                UseKarrasSigmas = false
            };
            _ops = new Ops(BackendType.GPUCompute);
            _scheduler = new EulerDiscreteScheduler(config);
            _scheduler.Ops = _ops;
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

        [Test]
        public void TestBetas() {
            CollectionAssert.AreEqual(ExpectedBetas, _scheduler.Betas, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestInitSigmas() {
            CollectionAssert.AreEqual(TestUtils.LoadFromFile("euler_discrete_test_sigmas"), _scheduler.Sigmas, new FloatArrayComparer(0.00001f));
            _scheduler.SetTimesteps(10);
            CollectionAssert.AreEqual(TestUtils.LoadFromFile("euler_discrete_test_sigmas_2"), _scheduler.Sigmas, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestInitNoiseSigma() {
            Assert.That(_scheduler.InitNoiseSigma, Is.EqualTo(14.648818f).Using(new FloatEqualityComparer(0.00001f)));
            _scheduler.SetTimesteps(10);
            Assert.That(_scheduler.InitNoiseSigma, Is.EqualTo(8.4500665f).Using(new FloatEqualityComparer(0.00001f)));
        }

        [Test]
        public void TestStepsOffset() {
            _scheduler.SetTimesteps(10);
            var expected = new int[] { 901, 801, 701, 601, 501, 401, 301, 201, 101, 1 };
            CollectionAssert.AreEqual(expected, _scheduler.Timesteps);
        }

        [Test]
        public void TestFullLoopNoNoise() {
            _scheduler.SetTimesteps(10);
            using var dummySamples = DummySamples;
            var sample = dummySamples;
            sample = _ops.Mul(_scheduler.InitNoiseSigma, sample);

            foreach (float t in _scheduler.Timesteps) {
                var residual = Model(sample, t);
                residual = _scheduler.ScaleModelInput(residual, t);
                var stepArgs = new StepArgs(residual, t, sample);
                sample = _scheduler.Step(stepArgs).PrevSample;
            }

            _ops.ExecuteCommandBufferAndClear();
            CollectionAssert.AreEqual(ExpectedOutput, sample.DownloadToArray(), new FloatArrayComparer(0.00001f));
        }

        private Tensor<float> Model(Tensor<float> sampleTensor, float t) {
            var result = _ops.Mul(sampleTensor, t / (t + 1));
            _ops.ExecuteCommandBufferAndClear();
            return result;
        }
    }
}