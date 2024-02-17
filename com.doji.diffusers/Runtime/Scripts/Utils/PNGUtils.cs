using System;
using System.IO;
using UnityEngine;

namespace Doji.AI.Diffusers {

    public static class PNGUtils {

        public static void SaveToDisk(this RenderTexture texture, string directory, Parameters parameters) {
            if (!Directory.Exists(directory)) {
                throw new ArgumentException($"The directory '{directory} does not exist");
            }

            string prompt = parameters.Prompt;
            var invalids = Path.GetInvalidFileNameChars();
            var fileName = string.Join("_", prompt.Split(invalids, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
            fileName = fileName[..Math.Min(fileName.Length, 60)];
            string filePath = Path.Combine(directory, $"{fileName}_{DateTime.Now:yyyyMMddHHmmss}.png");

            int width = texture.width;
            int height = texture.height;
            Texture2D tex = new Texture2D(width, height, TextureFormat.ARGB32, false);
            RenderTexture.active = texture;
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            File.WriteAllBytes(filePath, tex.EncodeToPNG());

            PNGUtils.AddMetadata(filePath, parameters);
        }

        /// <summary>
        /// Rewrites the given PNG file by adding the given json string to the
        /// PNG metadata under the 'description' keyword.
        /// </summary>
        public static void AddMetadata(string pngFilePath, Parameters data) {
            if (!File.Exists(pngFilePath)) {
                throw new FileNotFoundException($"The PNG file at {pngFilePath} was not found.");
            }

            string metaData = data.Serialize();
            Pngcs.PngCS.AddMetadata(pngFilePath, "parameters", metaData);    
        }
    }
}