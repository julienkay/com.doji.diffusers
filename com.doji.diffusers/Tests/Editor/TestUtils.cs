using System.IO;
using System;
using Unity.Sentis;
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

        public static TensorFloat LoadTensorFromFile(string fileName) {
            float[] data = LoadFromFile(fileName);
            return new TensorFloat(new TensorShape(data.Length), data);
        }

        public static TensorFloat LoadTensorFromFile(string fileName, TensorShape shape) {
            float[] data = LoadFromFile(fileName);
            return new TensorFloat(shape, data);
        }

        /// <summary>
        /// Dumps a tensor to a png file.
        /// </summary>
        public static void ToFile(string prompt, int width, int height, TensorFloat generated) {
            var tmp = RenderTexture.GetTemporary(width, height);
            TextureConverter.RenderToTexture(generated, tmp);
            var invalids = Path.GetInvalidFileNameChars();
            var fileName = string.Join("_", prompt.Split(invalids, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
            string filePath = $"{fileName}_{DateTime.Now.ToString("yyyyMMddHHmmss")}.png";
            Texture2D tex = new Texture2D(width, height, TextureFormat.ARGB32, false);
            RenderTexture.active = tmp;
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            File.WriteAllBytes(filePath, tex.EncodeToPNG());
        }
    }
}