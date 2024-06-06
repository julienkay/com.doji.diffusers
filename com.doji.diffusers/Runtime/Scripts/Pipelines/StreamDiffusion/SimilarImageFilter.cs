using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public class SimilarImageFilter : IDisposable {

        public float Threshold { get; private set; }
        public object PrevTensor { get; private set; }
        public object Cos { get; }
        public float MaxSkipFrame { get; private set; }
        public int SkipCount { get; private set; }

        private Ops _ops;

        public SimilarImageFilter(float threshold = 0.98f, float maxSkipFrame = 10f) {
            Threshold = threshold;
            PrevTensor = null;
            Cos = null; //torch.nn.CosineSimilarity(dim = 0, eps = 1e-6);
            MaxSkipFrame = maxSkipFrame;
            SkipCount = 0;
            _ops = new Ops(BackendType.GPUCompute);
        }

        public TensorFloat Execute(TensorFloat x) {
            return x;
            /*if (PrevTensor == null) {
                PrevTensor = x; //.detach().clone()
                return x;
            } else {
                cos_sim = cos(PrevTensor.reshape(-1), x.reshape(-1)).item()
                sample = random.uniform(0, 1)
                int skip_prob;
                if (Threshold >= 1) {
                    skip_prob = 0;
                } else {
                    skip_prob = max(0, 1 - (1 - cos_sim) / (1 - threshold));
                }

                // not skip frame
                if (skip_prob < sample) {
                    PrevTensor = x; //.detach().clone();
                    return x;
                // skip frame
                } else {
                    if (SkipCount > MaxSkipFrame) {
                        SkipCount = 0;
                        PrevTensor = x; //.detach().clone();
                        return x;
                    } else {
                        SkipCount += 1;
                        return null;
                    }
                }
            }*/
        }

        public void Dispose() {
            _ops?.Dispose();
        }
    }
}