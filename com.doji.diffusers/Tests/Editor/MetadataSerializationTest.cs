using NUnit.Framework;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class MetadataSerializationTest {

        [Test]
        public void Serialize() {
            Metadata metadata = new Metadata() {
                Model = "TestModel",
                Pipeline = "TestPipeline",
                Sampler = "TestSampler",
                Parameters = new Parameters {
                    Prompt = "TestPrompt"
                }
            };
            string serialized = metadata.Serialize();
            string packageVersion = System.Diagnostics.FileVersionInfo.GetVersionInfo(typeof(Metadata).Assembly.Location).ProductVersion;

            Assert.That(serialized, Contains.Substring(packageVersion));

            Metadata deserialized = Metadata.Deserialize(serialized);
            Assert.That(deserialized, Is.TypeOf(typeof(Metadata)));
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(deserialized.Comment, Is.Not.Null);
            Assert.AreEqual("TestPrompt", deserialized.Parameters.Prompt.ToString());
        }
    }
}