using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public class ModelOutput : Dictionary<int, Tensor> {
        public ModelOutput() : base() { }
        public void GetOutputs(Model model, IWorker worker) {
            int i = 0;
            foreach (var output in model.outputs) {
                this[i] = worker.PeekOutput(output);
                i++;
            }
        }
    }
}