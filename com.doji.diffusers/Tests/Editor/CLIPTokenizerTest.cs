using Doji.AI.Transformers;
using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class CLIPTokenizerTest : TestBase {

        /// <summary>
        /// Tests clip encoding without any padding and truncation
        /// </summary>
        [Test]
        public void TestCLIPEncodeSimple() {
            ClipTokenizer tokenizer = GetSDCLIPTokenizer();

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
            ClipTokenizer tokenizer = GetSDCLIPTokenizer();

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
            ClipTokenizer tokenizer = GetSDCLIPTokenizer();

            string prompt = "a cat";
            List<string> tokens = tokenizer.Tokenize(prompt);

            List<string> expected = new List<string>() { "a</w>", "cat</w>" };
            CollectionAssert.AreEqual(expected, tokens);
        }
    }
}