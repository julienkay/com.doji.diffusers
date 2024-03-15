using Doji.AI.Transformers;
using static Doji.AI.Diffusers.DiffusionPipeline;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class TestBase {

        /// <summary>
        /// Returns a CLIP tokenizer as used with Stable Diffusion 1.5
        /// </summary>
        protected ClipTokenizer GetSDCLIPTokenizer_1_5() {
            var model = DiffusionModel.SD_1_5;
            Vocab vocab = LoadVocab(model);
            string merges = LoadMerges(model);
            TokenizerConfig config = LoadTokenizerConfig(model);
            return new ClipTokenizer(vocab, merges, config);
        }

        protected ClipTokenizer GetSDCLIPTokenizer_2_1() {
            var model = DiffusionModel.SD_2_1;
            Vocab vocab = LoadVocab(model);
            string merges = LoadMerges(model);
            TokenizerConfig config = LoadTokenizerConfig(model);
            return new ClipTokenizer(vocab, merges, config);
        }
    }
}