using NUnit.Framework;
using System;
using System.Collections;
using Unity.Sentis;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class SchedulerBaseTest {
        public static IEnumerable StepsTestData {
            get {
                yield return new TestCaseData(typeof(DDIMScheduler)).Returns(Spacing.Leading);
                yield return new TestCaseData(typeof(PNDMScheduler)).Returns(Spacing.Leading);
                yield return new TestCaseData(typeof(EulerDiscreteScheduler)).Returns(Spacing.Linspace);
                yield return new TestCaseData(typeof(EulerAncestralDiscreteScheduler)).Returns(Spacing.Linspace);
            }
        }

        [Test]
        [TestCaseSource(nameof(StepsTestData))]
        public Spacing TestDefaultArgumentsNotInConfig(Type schedulerType) {
            using Scheduler scheduler = (Scheduler)Activator.CreateInstance(schedulerType, null, BackendType.GPUCompute);
            return scheduler.TimestepSpacing;
        }
    }
}