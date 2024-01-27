using Doji.AI.Transformers;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class TestBase {

        /// <summary>
        /// Returns a CLIP tokenizer as used with Stable Diffusion 1.5
        /// </summary>
        protected ClipTokenizer GetSDCLIPTokenizer() {
            string s = DiffusionModel.SD_1_5.Name;
            Vocab vocab = StableDiffusionPipeline.LoadVocab(s);
            string merges = StableDiffusionPipeline.LoadMerges(s);
            TokenizerConfig config = StableDiffusionPipeline.LoadTokenizerConfig(s);
            return new ClipTokenizer(vocab, merges, config);
        }
    }
}