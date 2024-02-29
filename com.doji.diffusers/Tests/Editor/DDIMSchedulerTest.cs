using NUnit.Framework;
using Unity.Sentis;
using static Doji.AI.Diffusers.Scheduler;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Tests for <see cref="DDIMScheduler"/>.
    /// </summary>
    public class DDIMSchedulerTest {

        private DDIMScheduler _scheduler;

        private float[] ExpectedBetas {
            get {
                return TestUtils.LoadFromFile("ddim_test_betas");
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
                return TestUtils.LoadFromFile("ddim_test_expected_output");
            }
        }

        private StepArgs _stepArgs = new StepArgs();

        [SetUp]
        public void SetUp() {
            var config = new SchedulerConfig() {
                BetaEnd = 0.02f,
                BetaSchedule = Schedule.Linear,
                BetaStart = 0.0001f,
                NumTrainTimesteps = 1000,
                ClipSample = true,
                SetAlphaToOne = false,
                StepsOffset = 1,
                PredictionType = Prediction.V_Prediction,
            };
            _scheduler = new DDIMScheduler(config);
        }

        [TearDown]
        public void TearDown() {
            _scheduler.Dispose();
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
        public void TestStepsOffset() {
            _scheduler.SetTimesteps(10);
            var expected = new int[] { 901, 801, 701, 601, 501, 401, 301, 201, 101, 1 };
            CollectionAssert.AreEqual(expected, _scheduler.Timesteps);
        }

        [Test]
        public void TestFullLoop() {
            _scheduler.SetTimesteps(10);
            using var dummySamples = DummySamples;
            var sample = dummySamples;

            foreach (int t in _scheduler.Timesteps) {
                var residual = Model(sample, t);
                _stepArgs.Set(residual, t, sample);
                sample = _scheduler.Step(_stepArgs).PrevSample;
                residual.Dispose();
            }

            sample.MakeReadable();
            CollectionAssert.AreEqual(ExpectedOutput, sample.ToReadOnlyArray(), new FloatArrayComparer(0.00001f));
        }

        private TensorFloat Model(TensorFloat sampleTensor, int t) {
            sampleTensor.MakeReadable();
            float[] sample = sampleTensor.ToReadOnlyArray();
            float[] result = new float[sample.Length];
            for (int i = 0; i < sample.Length; i++) {
                result[i] = sample[i] * ((float)t / (t + 1));
            }
            return new TensorFloat(sampleTensor.shape, result);
        }
    }
}