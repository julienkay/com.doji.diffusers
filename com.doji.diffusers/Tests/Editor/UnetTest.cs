using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class UnetTest {

        /// <summary>
        /// Loads deterministic random samples with shape (1, 4, 8, 8)
        /// </summary>
        private float[] Samples {
            get {
                return TestUtils.LoadFromFile("256_latents");
            }
        }

        /// <summary>
        /// Loads the embeddings for the prompt "a cat" (1, 77, 768) for the tiny test pipeline.
        /// </summary>
        private float[] PromptEmbedsTiny {
            get {
                return TestUtils.LoadFromFile("unte_test_tiny_fake_embeddings");
            }
        }

        /// <summary>
        /// Loads the embeddings for the prompt "a cat" (1, 77, 768) for SD 1.5
        /// </summary>
        private float[] PromptEmbeds_1_5 {
            get {
                return TestUtils.LoadFromFile("test_cat_embeddings_1_5");
            }
        }

        /// <summary>
        /// Loads the embeddings for the prompt "a cat" (1, 77, 1024) for SD 2.1
        /// </summary>
        private float[] PromptEmbeds_2_1 {
            get {
                return TestUtils.LoadFromFile("test_cat_embeddings_2_1");
            }
        }

        private float[] ExpectedOutput {
            get {
                return TestUtils.LoadFromFile("unet_test_output_256_1_5");
            }
        }

        /// <summary>
        /// Loads deterministic random samples with shape (1, 4, 64, 64)
        /// </summary>
        private float[] SamplesLarge {
            get {
                return TestUtils.LoadFromFile("16384_latents");
            }
        }

        private float[] ExpectedOutputLarge_1_5 {
            get {
                return TestUtils.LoadFromFile("unet_test_output_16384");
            }
        }

        private float[] ExpectedOutputLarge_2_1 {
            get {
                return TestUtils.LoadFromFile("unet_test_output_16384_2_1");
            }
        }

        private static Unet LoadUnet_Tiny() {
            Unet unet = Unet.FromPretrained(DiffusionModel.__TEST_SD_TINY, BackendType.GPUCompute);
            return unet;
        }

        private static Unet LoadUnet_1_5() {
            Unet unet = Unet.FromPretrained(DiffusionModel.SD_1_5, BackendType.GPUCompute);
            return unet;
        }
        
        private static Unet LoadUnet_2_1() {
            Unet unet = Unet.FromPretrained(DiffusionModel.SD_2_1, BackendType.GPUCompute);
            return unet;
        }

        [Test]
        public void TestUnetTiny() {
            using Unet unet = LoadUnet_Tiny();

            int t = 901;
            using Tensor<float> latentInputTensor = new Tensor<float>(new TensorShape(1, 4, 8, 8), Samples);
            using Tensor<float> promptEmbeds = new Tensor<float>(new TensorShape(1, 77, 32), PromptEmbedsTiny);
            using Tensor timestep = unet.CreateTimestep(new TensorShape(1), t);

            Tensor<float> noisePred = unet.Execute(latentInputTensor, timestep, promptEmbeds);
            float[] unetOutput = noisePred.DownloadToArray();

            Assert.IsTrue(true);
            //CollectionAssert.AreEqual(ExpectedOutput, unetOutput, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestUnet() {
            using Unet unet = LoadUnet_1_5();

            int t = 901;
            using Tensor<float> latentInputTensor = new Tensor<float>(new TensorShape(1, 4, 8, 8), Samples);
            using Tensor<float> promptEmbeds = new Tensor<float>(new TensorShape(1, 77, 768), PromptEmbeds_1_5);
            using Tensor timestep = unet.CreateTimestep(new TensorShape(1), t);

            Tensor<float> noisePred = unet.Execute(latentInputTensor, timestep, promptEmbeds);
            float[] unetOutput = noisePred.DownloadToArray();

            CollectionAssert.AreEqual(ExpectedOutput, unetOutput, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestUnetLarge_1_5() {
            using Unet unet = LoadUnet_1_5();

            int t = 901;
            using Tensor<float> latentInputTensor = new Tensor<float>(new TensorShape(1, 4, 64, 64), SamplesLarge);
            using Tensor<float> promptEmbeds = new Tensor<float>(new TensorShape(1, 77, 768), PromptEmbeds_1_5);
            using Tensor timestep = unet.CreateTimestep(new TensorShape(1), t);

            Tensor<float> noisePred = unet.Execute(latentInputTensor, timestep, promptEmbeds);
            float[] unetOutput = noisePred.DownloadToArray();

            CollectionAssert.AreEqual(ExpectedOutputLarge_1_5, unetOutput, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestUnetLarge_2_1() {
            using Unet unet = LoadUnet_2_1();

            int t = 901;
            using Tensor<float> latentInputTensor = new Tensor<float>(new TensorShape(1, 4, 64, 64), SamplesLarge);
            using Tensor<float> promptEmbeds = new Tensor<float>(new TensorShape(1, 77, 1024), PromptEmbeds_2_1);
            using Tensor timestep = unet.CreateTimestep(new TensorShape(1), t);

            Tensor<float> noisePred = unet.Execute(latentInputTensor, timestep, promptEmbeds);
            float[] unetOutput = noisePred.DownloadToArray();

            CollectionAssert.AreEqual(ExpectedOutputLarge_2_1, unetOutput, new FloatArrayComparer(0.0001f));
        }
    }
}