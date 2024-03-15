using System.Collections.Generic;
using UnityEditor;
using static Doji.AI.Diffusers.DiffusionPipeline;

namespace Doji.AI.Diffusers.Editor {

    public static class EditorUtils {

        private static HashSet<DiffusionModel> _downloads = new HashSet<DiffusionModel>();

        [InitializeOnLoadMethod]
        private static void InitializeOnLoad() {
            OnModelRequested -= Validate;
            OnModelRequested += Validate;
        }

        private static void Validate(DiffusionModel model) {
            if (IsModelAvailable(model)) {
                return;
            }
            if (InProgress(model)) {
                return;
            }
            if (!ShouldDownload(model)) {
                return;
            }

            EditorApplication.ExitPlaymode();
            DownloadUtils.DownloadModel(model);
        }

        /// <summary>
        /// Check whether user wants to download
        /// </summary>
        private static bool ShouldDownload(DiffusionModel model) {
            return EditorUtility.DisplayDialog(
               "com.doji.diffusers | Downloaded Model",
               "You are trying to use a diffusion model that is not yet downloaded to your machine.\n\n" +
               "Would you like to exit Play Mode and download the following model?\n\n" +
               $"{model.ModelId}\n\n" +
               "The download will happen in the background and might take a while.\n\n" +
               "Make sure to review the model's license at " + model.BaseUrl,
               "Download", "Cancel");
        }

        /// <summary>
        /// Is download for this model in progress?
        /// </summary>
        private static bool InProgress(DiffusionModel model) {
            if (_downloads == null) {
                return true;
            }
            return _downloads.Contains(model);
        }
    }
}
