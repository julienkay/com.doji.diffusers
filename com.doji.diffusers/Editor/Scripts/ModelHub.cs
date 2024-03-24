using System;
using System.IO;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using static Doji.AI.Diffusers.DiffusionPipeline;

namespace Doji.AI.Diffusers.Editor {

    public class ModelHub : SettingsProvider {

        internal static class Content {
            internal static readonly string SettingsRootTitle = "Project/Diffusers/Model Hub";
            internal static readonly string ModelsLabel = "Models";
        }

        public ModelHub(string path, SettingsScope scope = SettingsScope.User) : base(path, scope) { }

        public override void OnActivate(string searchContext, VisualElement rootElement) {
            base.OnActivate(searchContext, rootElement);
        }

        public override void OnGUI(string searchContext) {
            DrawModelHub();
        }

        private void DrawModelHub() {
            EditorGUILayout.Space(10);
            EditorGUI.indentLevel++;

            BeginSubHeading(Content.ModelsLabel);

            foreach (var model in DiffusionModel.ValidatedModels) {
                DrawModel(model);
            }

            EndSubHeading();

            EditorGUI.indentLevel--;
        }

        private void DrawModel(DiffusionModel model) {
            bool needsDownload = !ExistsInResources(model);
            if (needsDownload) {
                DrawDownload(model);
            } else {
                DrawModelInfo(model);
            }
        }

        private void DrawDownload(DiffusionModel model) {
            GUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(model.ModelId);

            if (GUILayout.Button("Download", GUILayout.Width(70))) {
                DownloadUtils.DownloadModel(model);
            }

            GUILayout.EndHorizontal();
        }

        private void DrawModelInfo(DiffusionModel model) {
            GUILayout.BeginHorizontal();

            EditorGUILayout.LabelField(model.ModelId);

            if (!ExistsInStreamingAssets(model)) {
                if (GUILayout.Button("Serialize to StreamingAssets")) {
                    EditorUtility.DisplayProgressBar("Model Hub - Serialize to StreamingAssets", $"Serializing '{model.ModelId}' to StreamingAssets...", 0.1f);
                    ConvertModel(model);
                    EditorUtility.ClearProgressBar();
                }
            }

            GUILayout.EndHorizontal();
        }

        /// <summary>
        /// Converts .onnx files of the given <paramref name="model"/> to .sentis
        /// and moves all the model files from Resources to the StreamingAssets folder.
        /// </summary>
        private void ConvertModel(DiffusionModel model) {
            string targetDir = Path.Combine(Application.streamingAssetsPath, model.Owner, model.ModelName, model.Revision);
            Directory.CreateDirectory(targetDir);

            foreach (var file in model) {
                MoveFile(file);
            }
        }

        private void MoveFile(ModelFile file) {
            if (!File.Exists(file.ResourcesFilePath)) {
                if (!file.Required) {
                    return;
                } else {
                    throw new Exception($"File at '{file.ResourcesFilePath}' not found.");
                }
            }

            // handle external weights file types: they get merged into .sentis files so do not need to be copied
            string ext = Path.GetExtension(file.ResourcesFilePath);
            if (ext == ".onnx_data" || ext == ".pb") {
                return;
            }

            var asset = Resources.Load(file.ResourcePath);
            if (asset == null) {
                throw new Exception($"Could not load asset at '{file.ResourcePath}'.");
            }

            Directory.CreateDirectory(Path.GetDirectoryName(file.StreamingAssetsPath));

            // .onnx to .sentis
            if (asset is ModelAsset) {
                ModelAsset modelAsset = asset as ModelAsset;
                Model model = new Model();
                ModelLoader.LoadModelDesc(modelAsset, ref model);
                ModelLoader.LoadModelWeights(modelAsset, ref model);
                string path = file.StreamingAssetsPath;
                ModelWriter.Save(path, model);
                AssetDatabase.Refresh();
                Resources.UnloadAsset(modelAsset);
            } else { // copy other files (text files like configs & model_index.json)
                File.Copy(file.ResourcesFilePath, file.StreamingAssetsPath, overwrite: true);
            }
            Resources.UnloadAsset(asset);
        }

        private void BeginSubHeading(string content, int spacing = 0) {
            GUILayout.Space(spacing);
            EditorGUILayout.LabelField(content, EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
        }

        private void EndSubHeading() {
            EditorGUI.indentLevel--;
            GUILayout.Space(10);
        }

        [SettingsProvider]
        public static SettingsProvider CreateMyCustomSettingsProvider() {
            return new ModelHub(Content.SettingsRootTitle, SettingsScope.Project);
        }
    }
}