using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine.Networking;
using System.Threading.Tasks;

namespace Doji.AI.Diffusers.Editor {

    public static class DownloadUtils {

        private static HashSet<DiffusionModel> _downloads = new HashSet<DiffusionModel>();

        /// <summary>
        /// Downloads the weights of a pretrained diffusion pipeline hosten od Hugging Face.
        /// </summary>
        /// <param name="model">the model to download</param>
        public async static Task DownloadModel(DiffusionModel model) {
            if (InProgress(model)) {
                return;
            }

            bool existsOnHF = await model.ExistsOnHF();
            if (!existsOnHF) {
                EditorUtility.DisplayDialog(
                    $"com.doji.diffusers | Unknown Model '{model.ModelId}'",
                    "The model you requested can not be found on Hugging Face.\n\n" +
                    "Check the spelling and that all required pipeline components are present in the repository.\n\n",
                    "OK"
                );
                return;
            }

            _downloads.Add(model);
            List<Task> tasks = new List<Task>();

            try {
                AssetDatabase.StartAssetEditing();
                foreach (var file in model) {
                    if (File.Exists(file.ResourcesFilePath)) {
                        continue;
                    }
                    Task t = DownloadFileAsync(file, model.ModelId);
                    tasks.Add(t);
                }
                await Task.WhenAll(tasks);
            } finally {
                _downloads.Remove(model);
                AssetDatabase.StopAssetEditing();
                AssetDatabase.Refresh();
            }
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

        /// <summary>
        /// Downloads the file from the given url.
        /// </summary>
        /// <param name="url">the source url</param>
        /// <param name="name">the name of the model this file belongs to</param>
        /// <param name="filePath">the path to save the downloaded file to</param>
        /// <param name="optional">if this is set to true, no error will be logged in case the file can not be found</param>
        /// <returns></returns>
        private async static Task DownloadFileAsync(ModelFile file, string name) {
            string filePath = file.ResourcesFilePath;
            string url = file.Url;
            string componentName = new FileInfo(filePath).Directory.Name;

            // first check if the file is available (some are optional)
            UnityWebRequest headRequest = UnityWebRequest.Head(url);
            var hr = headRequest.SendWebRequest();
            while (!hr.isDone) {
                await Task.Yield();
            }
            if (headRequest.responseCode == 404 && !file.Required) {
                return;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(filePath));

            UnityWebRequest wr = UnityWebRequest.Get(url);
            DownloadHandlerFile downloadHandler = new DownloadHandlerFile(filePath);
            wr.downloadHandler = downloadHandler;

            var asyncOp = wr.SendWebRequest();

            int dlID = Progress.Start($"Downloading {name} - {componentName}");
            Progress.RegisterCancelCallback(dlID, () => { return true; });

            bool canceled = false;

            while (!asyncOp.isDone) {
                if (Progress.GetStatus(dlID) == Progress.Status.Canceled) {
                    wr.Abort();
                    canceled = true;
                }
                Progress.Report(dlID, wr.downloadProgress, $"{Path.GetFileName(filePath)} download progress...");
                await Task.Yield();
            }
            Progress.Remove(dlID);

            if (canceled) {
                // remove partially downloaded file
                File.Delete(filePath);
                return;
            }

            if (wr.error != null || wr.result != UnityWebRequest.Result.Success) {
                File.Delete(filePath);
                EditorUtility.DisplayDialog(
                    "com.doji.diffusers | Download Error",
                    $"Downloading {url} failed.\n{wr.error}",
                    "OK"
                );
            }
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
               "The download will happen in the background and might take a while.",
               "Download", "Cancel");
        }

        /// <summary>
        /// Does this model point to a valid pipeline on Hugging Face?
        /// </summary>
        private async static Task<bool> ExistsOnHF(this DiffusionModel model) {
            UnityWebRequest headRequest = UnityWebRequest.Head(model.ModelIndex.Url);
            var hr = headRequest.SendWebRequest();
            while (!hr.isDone) {
                await Task.Yield();
            }

            if (headRequest.responseCode == 200) {
                return true;
            }
            
            return false;
        }
    }
}