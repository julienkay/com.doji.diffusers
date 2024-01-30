using Doji.AI.Transformers;
using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Test the <see cref="TextEncoder"/> of a <see cref="StableDiffusionPipeline"/>.
    /// Requires the models for runwayml/stable-diffusion-v1-5 to be downloaded.
    /// </summary>
    public class TextEncoderTest : TestBase {

        /// <summary>
        /// The expected (flattened) embedding for the prompt "a cat" with SD 1.5 defaults
        /// </summary>
        private float[] ExpectedEmbeddings {
            get {
                return TestUtils.LoadFromFile("encoder_test_cat_embeddings");
            }
        }

        /// <summary>
        /// The expected (flattened) embedding for an empty prompt with SD 1.5
        /// </summary>
        private float[] UnconditionalEmbeddings {
            get {
                return TestUtils.LoadFromFile("encoder_test_uncond_embeddings");
            }
        }

        [Test]
        public void TestEncode() {
            var model = StableDiffusionPipeline.LoadTextEncoder(DiffusionModel.SD_1_5.Name);

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

            Assert.AreEqual(ExpectedEmbeddings.Length, promptEmbeds.Length);

            //Assert.That(promptEmbeds, Is.EqualTo(expectedEmbedding).Within(0.0001f));
            CollectionAssert.AreEqual(ExpectedEmbeddings, promptEmbeds, new FloatArrayComparer(0.0001f));
        }

        [Test]
        public void TestEncodeUnconditional() {
            var model = StableDiffusionPipeline.LoadTextEncoder(DiffusionModel.SD_1_5.Name);

            ClipTokenizer tokenizer = GetSDCLIPTokenizer();

            var prompt = new List<string>() { "" };
            var inputIds = tokenizer.Encode<BatchInput>(
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

            CollectionAssert.AreEqual(UnconditionalEmbeddings, promptEmbeds, new FloatArrayComparer(0.0001f));
        }
    }
}