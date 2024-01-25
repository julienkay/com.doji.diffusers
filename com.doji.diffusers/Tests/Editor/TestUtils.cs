using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public static class TestUtils {
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
    }
}