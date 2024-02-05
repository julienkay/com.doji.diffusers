using System;
using Unity.Sentis;

namespace Doji.AI.Diffusers {

    public abstract class Scheduler : IDisposable {

        public SchedulerConfig Config { get; protected set; }

        public int NumTrainTimesteps { get => Config.NumTrainTimesteps; }
        public float BetaStart { get => Config.BetaStart; }
        public float BetaEnd { get => Config.BetaEnd; }
        public Schedule BetaSchedule { get => Config.BetaSchedule; }
        public bool SkipPrkSteps { get => Config.SkipPrkSteps; }
        public bool SetAlphaToOne { get => Config.SetAlphaToOne; }
        public int StepsOffset { get => Config.StepsOffset; }
        public float[] TrainedBetas { get => Config.TrainedBetas; }
        public Prediction PredictionType { get => Config.PredictionType; }
        public Spacing TimestepSpacing { get => Config.TimestepSpacing; }

        protected Ops _ops;

        public Scheduler(BackendType backend) {
            _ops = WorkerFactory.CreateOps(backend, null);
        }

        public virtual void Dispose() {
            _ops?.Dispose();
        }
    }
}