using Newtonsoft.Json;
using NUnit.Framework;
using System.Diagnostics;
using System.IO;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class PipelineConfigTest {

        [Test]
        public void BasicTest() {
            string cfgJson = "{\r\n  \"_class_name\": \"OnnxStableDiffusionPipeline\",\r\n  \"_diffusers_version\": \"0.6.0\",\r\n  \"feature_extractor\": [\r\n    \"transformers\",\r\n    \"CLIPFeatureExtractor\"\r\n  ],\r\n  \"safety_checker\": [\r\n    \"diffusers\",\r\n    \"OnnxRuntimeModel\"\r\n  ],\r\n  \"scheduler\": [\r\n    \"diffusers\",\r\n    \"PNDMScheduler\"\r\n  ],\r\n  \"text_encoder\": [\r\n    \"diffusers\",\r\n    \"OnnxRuntimeModel\"\r\n  ],\r\n  \"tokenizer\": [\r\n    \"transformers\",\r\n    \"CLIPTokenizer\"\r\n  ],\r\n  \"unet\": [\r\n    \"diffusers\",\r\n    \"OnnxRuntimeModel\"\r\n  ],\r\n  \"vae_decoder\": [\r\n    \"diffusers\",\r\n    \"OnnxRuntimeModel\"\r\n  ],\r\n  \"vae_encoder\": [\r\n    \"diffusers\",\r\n    \"OnnxRuntimeModel\"\r\n  ]\r\n}\r\n";
            var config = JsonConvert.DeserializeObject<PipelineConfig>(cfgJson);

            Assert.That(config, Is.Not.Null);
            Assert.That(config, Is.TypeOf<PipelineConfig>());
            Assert.That(config.DiffusersVersion, Is.EqualTo("0.6.0"));
            Assert.That(config.Scheduler.ClassName, Is.EqualTo("PNDMScheduler"));
        }

        [Test]
        public void SDXLConfigTest() {
            string cfgJson = "{\r\n  \"_class_name\": \"StableDiffusionXLPipeline\",\r\n  \"_diffusers_version\": \"0.19.0.dev0\",\r\n  \"force_zeros_for_empty_prompt\": true,\r\n  \"add_watermarker\": null,\r\n  \"scheduler\": [\r\n    \"diffusers\",\r\n    \"EulerDiscreteScheduler\"\r\n  ],\r\n  \"text_encoder\": [\r\n    \"transformers\",\r\n    \"CLIPTextModel\"\r\n  ],\r\n  \"text_encoder_2\": [\r\n    \"transformers\",\r\n    \"CLIPTextModelWithProjection\"\r\n  ],\r\n  \"tokenizer\": [\r\n    \"transformers\",\r\n    \"CLIPTokenizer\"\r\n  ],\r\n  \"tokenizer_2\": [\r\n    \"transformers\",\r\n    \"CLIPTokenizer\"\r\n  ],\r\n  \"unet\": [\r\n    \"diffusers\",\r\n    \"UNet2DConditionModel\"\r\n  ],\r\n  \"vae\": [\r\n    \"diffusers\",\r\n    \"AutoencoderKL\"\r\n  ]\r\n}\r\n";
            var config = JsonConvert.DeserializeObject<PipelineConfig>(cfgJson);
            Assert.That(config, Is.Not.Null);
            Assert.That(config, Is.TypeOf<StableDiffusionXLPipelineConfig>());
            Assert.That(config.DiffusersVersion, Is.EqualTo("0.19.0.dev0"));
            Assert.That(config.Scheduler.ClassName, Is.EqualTo("EulerDiscreteScheduler"));
            var xlConfig = config as StableDiffusionXLPipelineConfig;
            Assert.That(xlConfig.ForceZerosForEmptyPrompt, Is.True);
            Assert.That(xlConfig.AddWatermarker, Is.Null);
        }
    }
}