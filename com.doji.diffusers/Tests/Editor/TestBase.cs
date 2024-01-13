using Doji.AI.Transformers;
using System.IO;
using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class TestBase {

        public ModelAsset TextEncoder {
            get {
                string path = Path.Combine("text_encoder", "model");
                return Resources.Load<ModelAsset>(path);
            }
        }

        public Vocab Vocab {
            get {
                string path = Path.Combine("tokenizer", "vocab");
                TextAsset vocabFile = Resources.Load<TextAsset>(path);
                return Vocab.Deserialize(vocabFile.text);
            }
        }

        public string Merges {
            get {
                string path = Path.Combine("tokenizer", "merges");
                TextAsset mergesFile = Resources.Load<TextAsset>(path);
                return mergesFile.text;
            }
        }
    }
}