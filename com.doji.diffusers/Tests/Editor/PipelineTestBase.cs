using NUnit.Framework;
using Unity.Sentis;

namespace Doji.AI.Diffusers.Editor.Tests {

    public abstract class PipelineTestBase {

        /// <summary>
        /// Loads deterministic random samples with shape (1, 4, 8, 8)
        /// </summary>
        protected Tensor<float> Latents {
            get {
                return TestUtils.LoadTensorFromFile("256_latents", new TensorShape(1, 4, 8, 8));
            }
        }

        protected Tensor<float> LatentsLarge {
            get {
                return TestUtils.LoadTensorFromFile("16384_latents", new TensorShape(1, 4, 64, 64));
            }
        }

        protected float[] GetLatents(int i) {
            return TestUtils.LoadFromFile($"pipeline_test_latents_{i}");
        }

        protected float[] GetLatentsLarge(int i) {
            return TestUtils.LoadFromFile($"pipeline_test_latents_512_{i}");
        }

        protected DiffusionPipeline _pipeline;

        protected void SetUp(DiffusionModel model) {
            _pipeline = DiffusionPipeline.FromPretrained(model);

        }

        [OneTimeTearDown]
        public void TearDown() {
            _pipeline.Dispose();
        }
    }
}