using System;

namespace Doji.AI.Diffusers {

    public class ControlNetModel : IModel<ControlNetConfig>, IDisposable {
        public ControlNetConfig Config { get; }

        public void Dispose() {
 
        }
    }
}