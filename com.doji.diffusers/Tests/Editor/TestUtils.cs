using Unity.InferenceEngine;
using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public static class TestUtils {

        /// <summary>
        /// Loads the given file into a float array.
        /// Expects comma-separated values in a text file in Resources.
        /// </summary>
        public static float[] LoadFromFile(string fileName) {
            TextAsset file = Resources.Load<TextAsset>(fileName);
            if (file == null) {
                throw new System.ArgumentException($"File '{fileName}' not found.");
            }
            string text = file.text;
            string[] stringValues = text.Split(',');

            float[] floatValues = new float[stringValues.Length];

            // Parse each string element into a float and store it in the float array
            for (int i = 0; i < stringValues.Length; i++) {
                string value = stringValues[i];
                if (float.TryParse(value, out float result)) {
                    floatValues[i] = result;
                } else {
                    // Handle parsing error if needed
                    Debug.LogError($"Error parsing value at index {i}: {value}");
                }
            }
            return floatValues;
        }

        public static Tensor<float> LoadTensorFromFile(string fileName) {
            float[] data = LoadFromFile(fileName);
            return new Tensor<float>(new TensorShape(data.Length), data);
        }

        public static Tensor<float> LoadTensorFromFile(string fileName, TensorShape shape) {
            float[] data = LoadFromFile(fileName);
            return new Tensor<float>(shape, data);
        }

        /// <summary>
        /// Dumps a tensor to a png file.
        /// </summary>
        public static void ToFile(DiffusionPipeline sd, Tensor<float> generated) {
            var m = sd.GetMetadata();
            int width = m.Parameters.Width.Value;
            int height = m.Parameters.Height.Value;
            var tmp = RenderTexture.GetTemporary(width, height);
            TextureConverter.RenderToTexture(generated, tmp);
            PNGUtils.SaveToDisk(tmp, ".", m);
        }
    }
}