using System.Collections.Generic;
using Unity.InferenceEngine;

namespace Doji.AI.Diffusers {

    public class ModelOutput : Dictionary<int, Tensor> {

        public new Tensor this[int index] {
            get {
                // wrap around to allow for negative indexing
                index = (Count + (index % Count)) % Count;
                return base[index];
            }
            set {
                base[index] = value;
            }
        }

        public ModelOutput() : base() { }

        public void GetOutputs(Model model, Worker worker) {
            Clear();
            int i = 0;
            foreach (var output in model.outputs) {
                this[i++] = worker.PeekOutput(output.name);
            }
        }
    }
}