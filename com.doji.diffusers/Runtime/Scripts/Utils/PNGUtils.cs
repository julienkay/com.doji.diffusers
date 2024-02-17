using System.IO;

namespace Doji.AI.Diffusers {

    public static class PNGUtils {

        public static Parameters GetParameters(
            StableDiffusionPipeline pipeline,
            string modelName,
            string prompt,
            int height = 512,
            int width = 512,
            int numInferenceSteps = 50,
            float guidanceScale = 7.5f,
            string negativePrompt = null,
            float eta = 0.0f)
        {
            return new Parameters() {
                PackageVersion = System.Reflection.Assembly.GetExecutingAssembly().GetName().Version,
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                Steps = numInferenceSteps,
                Sampler = pipeline.Scheduler.GetType().Name,
                CfgScale = guidanceScale,
                Width = width,
                Height = height,
                Model = modelName,
                Eta = eta
            };
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