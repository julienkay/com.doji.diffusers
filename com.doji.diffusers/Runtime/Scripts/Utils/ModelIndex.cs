using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;

public partial class ModelIndex {

    [JsonProperty("_class_name")]
    public virtual string ClassName { get; set; }

    [JsonProperty("_diffusers_version")]
    public virtual string DiffusersVersion { get; set; }

    [JsonProperty("force_zeros_for_empty_prompt")]
    public virtual bool ForceZerosForEmptyPrompt { get; set; }

    [JsonProperty("add_watermarker")]
    public virtual object AddWatermarker { get; set; }

    public virtual IDictionary<string, string[]> Components { get; set; }

    public static ModelIndex Deserialize(string s) {
        return JsonConvert.DeserializeObject<ModelIndex>(s, new ModelIndexConverter());
    }
}

public class ModelIndexConverter : JsonConverter<ModelIndex> {
    public override ModelIndex ReadJson(JsonReader reader, Type objectType, ModelIndex existingValue, bool hasExistingValue, JsonSerializer serializer) {
        JObject obj = JObject.Load(reader);
        var modelIndex = new ModelIndex();

        // Deserialize properties with [JsonProperty] attributes
        serializer.Populate(obj.CreateReader(), modelIndex);

        modelIndex.Components = new Dictionary<string, string[]>();

        foreach (var property in obj.Properties()) {
            if (property.Name != "_class_name" && property.Name != "_diffusers_version" &&
                property.Name != "force_zeros_for_empty_prompt" && property.Name != "add_watermarker") {
                var value = property.Value as JArray;
                modelIndex.Components[property.Name] = value?.ToObject<string[]>();
            }
        }

        return modelIndex;
    }

    public override void WriteJson(JsonWriter writer, ModelIndex value, JsonSerializer serializer) {
        throw new NotImplementedException();
    }
}