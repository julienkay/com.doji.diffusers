using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public abstract class Scheduler : IDisposable {

        public SchedulerConfig Config { get; protected set; }

        protected Ops _ops;

        public Scheduler(BackendType backend) {
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        public virtual void Dispose() {
            _ops?.Dispose();
        }
    }
}