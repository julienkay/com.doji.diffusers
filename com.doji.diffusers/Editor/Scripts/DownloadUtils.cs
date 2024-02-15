using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine.Networking;
using System.Threading.Tasks;

namespace Doji.AI.Diffusers.Editor {

    public static class DownloadUtils {

        private static HashSet<DiffusionModel> _downloads = new HashSet<DiffusionModel>();

        public async static void DownloadModel(DiffusionModel model) {
            if (InProgress(model)) {
                return;
            }

            _downloads.Add(model);
            List<Task> tasks = new List<Task>();
            AssetDatabase.StartAssetEditing();

            try {
                foreach ((string url, string filePath, bool optional) in model) {
                    if (File.Exists(filePath)) {
                        continue;
                    }
                    Task t = DownloadModelAsync(url, model.Name, filePath, optional);
                    tasks.Add(t);
                }
                await Task.WhenAll(tasks);
            } finally {
                _downloads.Remove(model);
                AssetDatabase.StopAssetEditing();
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
        private async static Task DownloadModelAsync(string url, string name, string filePath, bool optional) {
            string fileName = Path.GetFileName(filePath);
            Directory.CreateDirectory(Path.GetDirectoryName(filePath));

            UnityWebRequest wr = UnityWebRequest.Get(url);
            DownloadHandlerFile downloadHandler = new DownloadHandlerFile(filePath);
            wr.downloadHandler = downloadHandler;

            var asyncOp = wr.SendWebRequest();

            int dlID = Progress.Start($"Downloading {name} - {fileName}");
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
                return;
            }

            if (wr.responseCode == 404 && optional) {
                File.Delete(filePath);
                return;
            } else if (wr.error != null || wr.result != UnityWebRequest.Result.Success) {
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
               $"{model.Name}\n\n" +
               "The download will happen in the background and might take a while.",
               "Download", "Cancel");
        }

    }
}