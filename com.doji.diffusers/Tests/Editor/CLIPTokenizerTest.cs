using Doji.AI.Transformers;
using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class CLIPTokenizerTest : TestBase {

        [Test]
        public void TestCLIPEncode() {
            ClipTokenizer tokenizer = new ClipTokenizer(Vocab, Merges);

            string prompt = "a cat";
            List<int> inputIds = tokenizer.Encode(
                prompt,
                padding: Padding.MaxLength,
                maxLength: tokenizer.ModelMaxLength,
                truncation: Truncation.LongestFirst
            ).InputIds;

            Debug.Log(string.Join(", ", inputIds));

            List<int> expected = new List<int>() { 49406, 320, 2368, 49407 };
            CollectionAssert.AreEqual(expected, inputIds);
        }

        [Test]
        public void TestCLIPTokenize() {
            ClipTokenizer tokenizer = new ClipTokenizer(Vocab, Merges);

            string prompt = "a cat";
            List<string> tokens = tokenizer.Tokenize(prompt);

            Debug.Log(string.Join(", ", tokens));

            List<string> expected = new List<string>() { "a</w>", "cat</w>" };
            CollectionAssert.AreEqual(expected, tokens);
        }
    }
}