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
        private float[] ExpectedEmbeddings_1_5 {
            get {
                return TestUtils.LoadFromFile("test_cat_embeddings_1_5");
            }
        }

        /// <summary>
        /// The expected (flattened) embedding for the prompt "a cat" with SD 2.1 defaults
        /// </summary>
        private float[] ExpectedEmbeddings_2_1 {
            get {
                return TestUtils.LoadFromFile("test_cat_embeddings_2_1");
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
        public void TestEncode_1_5() {
            var model = DiffusionPipeline.LoadTextEncoder(DiffusionModel.SD_1_5);

            ClipTokenizer tokenizer = GetSDCLIPTokenizer_1_5();

            string prompt = "a cat";
            var inputIds = tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            ).InputIds;

            using Tensor<int> tokens = new Tensor<int>(new TensorShape(1, inputIds.Count()), inputIds.ToArray());
            using TextEncoder textEncoder = new TextEncoder(model, null);
            Tensor<float> output = textEncoder.Execute(tokens)[0] as Tensor<float>;
            output.ReadbackAndClone();
            float[] promptEmbeds = output.DownloadToArray();

            CollectionAssert.AreEqual(ExpectedEmbeddings_1_5, promptEmbeds, new FloatArrayComparer(0.0001f));
        }

        [Test]
        public void TestEncode_2_1() {
            var model = DiffusionPipeline.LoadTextEncoder(DiffusionModel.SD_2_1);

            ClipTokenizer tokenizer = GetSDCLIPTokenizer_2_1();

            string prompt = "a cat";
            var inputIds = tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            ).InputIds;

            using Tensor<int> tokens = new Tensor<int>(new TensorShape(1, inputIds.Count()), inputIds.ToArray());
            using TextEncoder textEncoder = new TextEncoder(model, null);
            Tensor<float> output = textEncoder.Execute(tokens)[0] as Tensor<float>;
            output.ReadbackAndClone();
            float[] promptEmbeds = output.DownloadToArray();

            CollectionAssert.AreEqual(ExpectedEmbeddings_2_1, promptEmbeds, new FloatArrayComparer(0.0001f));
        }

        [Test]
        public void TestEncodeUnconditional() {
            var model = DiffusionPipeline.LoadTextEncoder(DiffusionModel.SD_1_5);

            ClipTokenizer tokenizer = GetSDCLIPTokenizer_1_5();

            var prompt = new List<string>() { "" };
            var inputIds = tokenizer.Encode<BatchInput>(
                prompt,
                padding: Padding.MaxLength,
                maxLength: tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            ).InputIds;

            using Tensor<int> tokens = new Tensor<int>(new TensorShape(1, inputIds.Count()), inputIds.ToArray());
            using TextEncoder textEncoder = new TextEncoder(model, null);
            Tensor<float> output = textEncoder.Execute(tokens)[0] as Tensor<float>;
            output.ReadbackAndClone();
            float[] promptEmbeds = output.DownloadToArray();

            CollectionAssert.AreEqual(UnconditionalEmbeddings, promptEmbeds, new FloatArrayComparer(0.0001f));
        }
    }
}