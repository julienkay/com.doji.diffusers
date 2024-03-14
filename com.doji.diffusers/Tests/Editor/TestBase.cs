using Doji.AI.Transformers;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class TestBase {

        /// <summary>
        /// Returns a CLIP tokenizer as used with Stable Diffusion 1.5
        /// </summary>
        protected ClipTokenizer GetSDCLIPTokenizer_1_5() {
            string s = DiffusionModel.SD_1_5.ModelId;
            Vocab vocab = DiffusionPipeline.LoadVocab(s, "tokenizer");
            string merges = DiffusionPipeline.LoadMerges(s, "tokenizer");
            TokenizerConfig config = DiffusionPipeline.LoadTokenizerConfig(s, "tokenizer");
            return new ClipTokenizer(vocab, merges, config);
        }

        protected ClipTokenizer GetSDCLIPTokenizer_2_1() {
            string s = DiffusionModel.SD_2_1.ModelId;
            Vocab vocab = DiffusionPipeline.LoadVocab(s, "tokenizer");
            string merges = DiffusionPipeline.LoadMerges(s, "tokenizer");
            TokenizerConfig config = DiffusionPipeline.LoadTokenizerConfig(s, "tokenizer");
            return new ClipTokenizer(vocab, merges, config);
        }
    }
}