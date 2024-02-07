using NUnit.Framework;
using Unity.Sentis;

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
        /// Loads the embeddings for the prompt "a cat" (1, 77, 768)
        /// </summary>
        private float[] PromptEmbeds {
            get {
                return TestUtils.LoadFromFile("test_cat_embeddings");
            }
        }

        private float[] ExpectedOutput {
            get {
                return TestUtils.LoadFromFile("unet_test_output_256");
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

        private float[] ExpectedOutputLarge {
            get {
                return TestUtils.LoadFromFile("unet_test_output_16384");
            }
        }
        private static Unet LoadUnet() {
            var model = StableDiffusionPipeline.LoadUnet(DiffusionModel.SD_1_5.Name);
            return new Unet(model);
        }

        [Test]
        public void TestUnet() {
            using Unet unet = LoadUnet();

            int t = 901;
            using TensorFloat latentInputTensor = new TensorFloat(new TensorShape(1, 4, 8, 8), Samples);
            using TensorFloat promptEmbeds = new TensorFloat(new TensorShape(1, 77, 768), PromptEmbeds);
            using TensorInt timestep = new TensorInt(new TensorShape(1), ArrayUtils.Full(1, t));

            TensorFloat noisePred = unet.ExecuteModel(latentInputTensor, timestep, promptEmbeds);
            noisePred.MakeReadable();
            float[] unetOutput = noisePred.ToReadOnlyArray();

            CollectionAssert.AreEqual(ExpectedOutput, unetOutput, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestUnetLarge() {
            using Unet unet = LoadUnet();

            int t = 901;
            using TensorFloat latentInputTensor = new TensorFloat(new TensorShape(1, 4, 64, 64), SamplesLarge);
            using TensorFloat promptEmbeds = new TensorFloat(new TensorShape(1, 77, 768), PromptEmbeds);
            using TensorInt timestep = new TensorInt(new TensorShape(1), ArrayUtils.Full(1, t));

            TensorFloat noisePred = unet.ExecuteModel(latentInputTensor, timestep, promptEmbeds);
            noisePred.MakeReadable();
            float[] unetOutput = noisePred.ToReadOnlyArray();

            CollectionAssert.AreEqual(ExpectedOutputLarge, unetOutput, new FloatArrayComparer(0.00001f));
        }
    }
}