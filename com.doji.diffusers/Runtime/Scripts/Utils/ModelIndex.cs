using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;

namespace Doji.AI.Diffusers {

    [JsonConverter(typeof(PipelineConfigConverter))]
    public partial class PipelineConfig {

        [JsonProperty("_class_name")]
        public virtual string ClassName { get; set; }

        [JsonProperty("_diffusers_version")]
        public virtual string DiffusersVersion { get; set; }

        [JsonProperty("force_zeros_for_empty_prompt")]
        public virtual bool ForceZerosForEmptyPrompt { get; set; }

        [JsonProperty("add_watermarker")]
        public virtual object AddWatermarker { get; set; }

        [JsonProperty("requires_aesthetics_score")]
        public virtual bool RequiresAestheticsScore { get; set; }

        public virtual IDictionary<string, string[]> Components { get; set; }
    }

    public class PipelineConfigConverter : JsonConverter<PipelineConfig> {
        public override PipelineConfig ReadJson(JsonReader reader, Type objectType, PipelineConfig existingValue, bool hasExistingValue, JsonSerializer serializer) {
            JObject obj = JObject.Load(reader);
            var config = new PipelineConfig();

            // Deserialize properties with [JsonProperty] attributes
            serializer.Populate(obj.CreateReader(), config);

            config.Components = new Dictionary<string, string[]>();

            foreach (var property in obj.Properties()) {
                if (property.Name != "_class_name" && property.Name != "_diffusers_version" &&
                    property.Name != "force_zeros_for_empty_prompt" && property.Name != "add_watermarker" &&
                    property.Name != "requires_aesthetics_score") {
                    var value = property.Value as JArray;
                    config.Components[property.Name] = value?.ToObject<string[]>();
                }
            }

            return config;
        }

        public override void WriteJson(JsonWriter writer, PipelineConfig value, JsonSerializer serializer) {
            throw new NotImplementedException();
        }
    }
}