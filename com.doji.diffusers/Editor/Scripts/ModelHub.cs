using System;
using System.IO;
using System.Linq;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using static Doji.AI.Diffusers.DiffusionPipeline;

namespace Doji.AI.Diffusers.Editor {

    public class ModelHub : SettingsProvider {

        internal static class Content {
            internal static readonly string SettingsRootTitle = "Project/Diffusers/Model Hub";
            internal static readonly string ModelsLabel = "Base Models";
            internal static readonly string CustomLabel = "Custom Models";
            internal static readonly string ModelId = "Model ID";
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

            DrawBaseModels();
            DrawDownloadCustomDownload();

            EditorGUI.indentLevel--;
        }

        private string _modelInput;

        private void DrawDownloadCustomDownload() {
            BeginSubHeading(Content.CustomLabel);

            const string DEFAULT = "<enter model id (e.g. 'stabilityai/sdxl-turbo')>";
            GUILayout.BeginHorizontal();
            GUILayout.BeginVertical();

            bool valid = !string.IsNullOrEmpty(_modelInput) && _modelInput != DEFAULT;

            if (!valid) {
                var style = new GUIStyle(EditorStyles.textField);
                style.normal.textColor = Color.grey;
                style.active.textColor = Color.grey;
                style.focused.textColor = Color.grey;
                style.hover.textColor = Color.grey;
                _modelInput = EditorGUILayout.TextField(Content.ModelId, DEFAULT, style);
            } else {
                _modelInput = EditorGUILayout.TextField(Content.ModelId, _modelInput);
            }

            valid = !string.IsNullOrEmpty(_modelInput) && _modelInput != DEFAULT;

            if (!EditorGUIUtility.editingTextField && _modelInput.Split("/").Count() != 2) {
                EditorGUILayout.HelpBox("model id is in an invalid format. (Use 'author/repoName')", MessageType.Error);
                valid = false;
            }
            GUILayout.EndVertical();

            EditorGUI.BeginDisabledGroup(!valid);
            if (GUILayout.Button("Download", GUILayout.Width(70))) {
                var model = new DiffusionModel(_modelInput);
                // TODO: display message if model is already on users machine
                if (ExistsInResources(model)) {
                    EditorUtility.DisplayDialog("Model Hub", $"The model '{_modelInput}' is already downloaded.", "OK");
                } else {
                    _ = DownloadUtils.DownloadModel(model);
                }
            }
            EditorGUI.EndDisabledGroup();

            GUILayout.EndHorizontal();

            EndSubHeading();
        }

        private void DrawBaseModels() {
            BeginSubHeading(Content.ModelsLabel);

            foreach (var model in DiffusionModel.ValidatedModels) {
                DrawModel(model);
            }

            EndSubHeading();
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
                _ = DownloadUtils.DownloadModel(model);
            }

            GUILayout.EndHorizontal();
        }

        private void DrawModelInfo(DiffusionModel model) {
            GUILayout.BeginHorizontal();

            EditorGUILayout.LabelField(model.ModelId);

            if (GUILayout.Button("Convert/Quantize")) {
                ConversionWindow.ShowWindow(model.ModelId);
            }

            GUILayout.EndHorizontal();
        }

        /// <summary>
        /// Converts .onnx files of the given <paramref name="model"/> to .sentis
        /// and moves all the model files from Resources to the StreamingAssets folder.
        /// It can also optionally quantize the model using the given data type.
        /// </summary>
        internal static void ConvertModel(DiffusionModel model, QuantizationType? quantizationType = null) {
            string targetDir = Path.Combine(Application.streamingAssetsPath, model.Owner, model.ModelName);
            Directory.CreateDirectory(targetDir);

            foreach (var file in model) {
                ConvertFile(file, quantizationType);
            }

            if (quantizationType != null) {
                // rename main folder to disambiguate between differently quantized models
                Directory.Move(targetDir, $"{targetDir}_{quantizationType.ToString().ToLower()}");
                File.Delete($"{targetDir}.meta");
            }
        }

        /// <summary>
        /// Moves the file to the StreamingAssets folder and if it's an .onnx file, converts it to .sentis format.
        /// </summary>
        private static void ConvertFile(ModelFile file, QuantizationType? quantizationType = null) {
            bool shouldQuantize = quantizationType != null;

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
                Model model = ModelLoader.Load(modelAsset);
                string path = file.StreamingAssetsPath;
                if (shouldQuantize) {
                    ModelQuantizer.QuantizeWeights(quantizationType.Value, ref model);
                }
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