using NUnit.Framework;
using System.IO;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class UnetTest {


        /// <summary>
        /// Loads deterministic random samples with shape (1, 3, 8, 8)
        /// </summary>
        private float[] Samples {
            get {
                return TestUtils.LoadFromFile("unet_test_256_latents");
            }
        }

        /// <summary>
        /// Loads the embeddings for the prompt "a cat" (1, 77, 768)
        /// </summary>
        private float[] PromptEmbeds {
            get {
                return TestUtils.LoadFromFile("encoder_test_last_hidden_state");
            }
        }

        private float[] ExpectedOutput {
            get {
                return TestUtils.LoadFromFile("unet_test_output");
            }
        }

        private static Unet LoadUnet() {
            string path = Path.Combine("unet", "model");
            var model = Resources.Load<ModelAsset>(path);
            if (model == null) {
                throw new FileNotFoundException($"The model filed for the text encoder was not found at: '{path}'");
            }
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
    }
}