using Doji.AI.Transformers;
using NUnit.Framework;
using System.Collections.Generic;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class CLIPTokenizerTest : TestBase {

        /// <summary>
        /// Tests clip encoding without any padding and truncation
        /// </summary>
        [Test]
        public void TestCLIPEncodeSimple() {
            ClipTokenizer tokenizer = GetSDCLIPTokenizer_1_5();

            string prompt = "a cat";
            var inputIds = tokenizer.Encode(prompt).InputIds;

            List<int> expected = new List<int>() { 49406, 320, 2368, 49407 };
            CollectionAssert.AreEqual(expected, inputIds);
        }

        /// <summary>
        /// Tests clip encoding with the same padding and truncation settings
        /// as in the Stable Diffusion pipelines.
        /// </summary>
        [Test]
        public void TestCLIPEncode() {
            ClipTokenizer tokenizer = GetSDCLIPTokenizer_1_5();

            string prompt = "a cat";
            var inputIds = tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            ).InputIds;

            List<int> expected = new List<int>() {
                49406,   320,  2368, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407
            };
            CollectionAssert.AreEqual(expected, inputIds);
        }

        [Test]
        public void TestCLIPTokenize() {
            ClipTokenizer tokenizer = GetSDCLIPTokenizer_1_5();

            string prompt = "a cat";
            List<string> tokens = tokenizer.Tokenize(prompt);

            List<string> expected = new List<string>() { "a</w>", "cat</w>" };
            CollectionAssert.AreEqual(expected, tokens);
        }

        /// <summary>
        /// Tests clip encoding with an empty input.
        /// </summary>
        [Test]
        public void TestCLIPEncodeEmpty() {
            ClipTokenizer tokenizer = GetSDCLIPTokenizer_1_5();

            string prompt = "";
            var inputIds = tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            ).InputIds;

            List<int> expected = new List<int>() {
                49406, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                49407, 49407, 49407, 49407, 49407
            };
            CollectionAssert.AreEqual(expected, inputIds);
        }
    }
}