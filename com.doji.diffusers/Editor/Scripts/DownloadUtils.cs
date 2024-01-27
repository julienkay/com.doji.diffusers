using System.Collections.Generic;
using System.IO;
using UnityEditor;
using static Doji.AI.Diffusers.StableDiffusionPipeline;
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

            foreach ((string url, string filePath, bool optional) in model) {
                if (!File.Exists(filePath)) {
                    Task t = DownloadModelAsync(model, url, filePath, optional);
                    tasks.Add(t);
                }
            }
            await Task.WhenAll(tasks);

            _downloads.Remove(model);
        }

        /// <summary>
        /// Is download for this name in progress?
        /// </summary>
        private static bool InProgress(DiffusionModel model) {
            if (_downloads == null) {
                return true;
            }
            return _downloads.Contains(model);
        }

        private async static Task DownloadModelAsync(DiffusionModel model, string url, string filePath, bool optional) {
            string name = model.Name;
            string fileName = Path.GetFileName(filePath);

            UnityWebRequest wr = UnityWebRequest.Get(url);
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

            byte[] data = asyncOp.webRequest.downloadHandler.data;

            if (wr.responseCode == 404 && optional) {
                return;
            } else if (wr.error != null || data == null || data.Length == 0) {
                EditorUtility.DisplayDialog(
                    "com.doji.diffusers | Download Error",
                    $"Downloading {url} failed.\n{wr.error}",
                    "OK"
                );
            } else {
                Directory.CreateDirectory(Path.GetDirectoryName(filePath));
                File.WriteAllBytes(
                    filePath,
                    data
                );
                AssetDatabase.Refresh();
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