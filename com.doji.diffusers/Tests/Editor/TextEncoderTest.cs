using Doji.AI.Transformers;
using NUnit.Framework;
using System.IO;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Test the <see cref="TextEncoder"/> of a <see cref="StableDiffusionPipeline"/>.
    /// Requires the models for runwayml/stable-diffusion-v1-5 to be downloaded.
    /// </summary>
    public class TextEncoderTest : TestBase {

        /// <summary>
        /// Loads the expected (flattened) embedding for the prompt "a cat" with SD 1.5 defaults from a text file
        /// </summary>
        private float[] ExpectedEmbedding {
            get {
                return TestUtils.LoadFromFile("encoder_test_last_hidden_state");
            }
        }

        [Test]
        public void TestEncode() {
            string path = Path.Combine("text_encoder", "model");
            var model = Resources.Load<ModelAsset>(path);
            if (model == null) {
                throw new FileNotFoundException($"The model filed for the text encoder was not found at: '{path}'");
            }

            ClipTokenizer tokenizer = GetSDCLIPTokenizer();

            string prompt = "a cat";
            var inputIds = tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            ).InputIds;

            using TensorInt tokens = new TensorInt(new TensorShape(1, inputIds.Count()), inputIds.ToArray());
            using TextEncoder textEncoder = new TextEncoder(model);
            TensorFloat output = textEncoder.ExecuteModel(tokens);
            output.MakeReadable();
            float[] promptEmbeds = output.ToReadOnlyArray();

            Debug.Log(string.Join(", ", promptEmbeds));
            Assert.AreEqual(ExpectedEmbedding.Length, promptEmbeds.Length);

            //Assert.That(promptEmbeds, Is.EqualTo(expectedEmbedding).Within(0.0001f));
            CollectionAssert.AreEqual(ExpectedEmbedding, promptEmbeds, new FloatArrayComparer(0.0001f));
        }
    }
}