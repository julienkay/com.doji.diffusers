using NUnit.Framework;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class MetadataSerializationTest {

        [Test]
        public void TestRoundtrip() {
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

        [Test]
        public void TestPackageVersion() {
            string json = "{\"comment\":\"This image was generated using https://github.com/julienkay/com.doji.diffusers\",\"package_version\":\"0.1.0\",\"model\":\"TestModel\",\"pipeline\":\"TestPipeline\",\"sampler\":\"TestSampler\",\"parameters\":{}}";
            Metadata deserialized = Metadata.Deserialize(json);
            Assert.That(deserialized, Is.TypeOf(typeof(Metadata)));
            Assert.That(deserialized, Is.Not.Null);
            Assert.That(deserialized.PackageVersion, Is.Not.Null);
            Assert.AreEqual("0.1.0", deserialized.PackageVersion);
            string packageVersion = System.Diagnostics.FileVersionInfo.GetVersionInfo(typeof(Metadata).Assembly.Location).ProductVersion;
        }
    }
}