using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;

namespace Doji.AI.Diffusers {

    public partial class StableDiffusionXLPipelineConfig : PipelineConfig {
        [JsonProperty("force_zeros_for_empty_prompt")]
        public bool ForceZerosForEmptyPrompt { get; set; }

        [JsonProperty("add_watermarker")]
        public object AddWatermarker { get; set; }

        [JsonProperty("requires_aesthetics_score")]
        public virtual bool RequiresAestheticsScore { get; set; }
    }

    /// <summary>
    /// The config for a pipeline (usually "model_index.json")
    /// </summary>
    [JsonConverter(typeof(PipelineConfigConverter))]
    public class PipelineConfig {

        [JsonProperty("_class_name")]
        public string ClassName { get; set; }

        [JsonProperty("_diffusers_version")]
        public string DiffusersVersion { get; set; }

        [JsonProperty("feature_extractor")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent FeatureExtractor { get; set; }

        [JsonProperty("safety_checker")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent SafetyChecker { get; set; }

        [JsonProperty("scheduler")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent Scheduler { get; set; }

        [JsonProperty("text_encoder")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent TextEncoder { get; set; }

        [JsonProperty("tokenizer")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent Tokenizer { get; set; }

        [JsonProperty("unet")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent Unet { get; set; }

        [JsonProperty("vae_decoder")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent VaeDecoder { get; set; }

        [JsonProperty("vae_encoder")]
        [JsonConverter(typeof(PipelineComponentConverter))]
        public PipelineComponent VaeEncoder { get; set; }
    }

    public class PipelineConfigConverter : JsonConverter<PipelineConfig> {
        public override PipelineConfig ReadJson(JsonReader reader, Type objectType, PipelineConfig existingValue, bool hasExistingValue, JsonSerializer serializer) {
            var jsonObject = JObject.Load(reader);
            // Read the "_class_name" property to determine the subclass type
            var className = jsonObject["_class_name"].ToString();
            PipelineConfig config = className switch {
                "StableDiffusionXLPipeline" => new StableDiffusionXLPipelineConfig(),
                _ => new PipelineConfig(),
            };

            // Populate the object using the default serializer
            serializer.Populate(jsonObject.CreateReader(), config);

            return config;
        }

        public override void WriteJson(JsonWriter writer, PipelineConfig value, JsonSerializer serializer) {
            serializer.Serialize(writer, value);
        }
    }

    public struct PipelineComponent {
        public string Library { get; set; }
        public string ClassName { get; set; }
    }

    public class PipelineComponentConverter : JsonConverter<PipelineComponent> {
        public override PipelineComponent ReadJson(JsonReader reader, Type objectType, PipelineComponent existingValue, bool hasExistingValue, JsonSerializer serializer) {
            // Load the JSON array
            JArray array = JArray.Load(reader);

            // Ensure the array contains exactly two elements
            if (array.Count != 2) {
                throw new JsonSerializationException("Expected an array with exactly two elements.");
            }

            // Return a Component object with the values from the array
            return new PipelineComponent {
                Library = (string)array[0],
                ClassName = (string)array[1]
            };
        }

        public override void WriteJson(JsonWriter writer, PipelineComponent value, JsonSerializer serializer) {
            // Serialize PipelineComponent as an array with two values
            writer.WriteStartArray();
            writer.WriteValue(value.Library);
            writer.WriteValue(value.ClassName);
            writer.WriteEndArray();
        }
    }
}