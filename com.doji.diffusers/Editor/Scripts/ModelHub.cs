using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using static Doji.AI.Diffusers.StableDiffusionPipeline;

namespace Doji.AI.Diffusers.Editor {

    public class ModelHub : SettingsProvider {

        internal static class Content {
            internal static readonly string SettingsRootTitle = "Project/Doji/Diffusers/Model Hub";
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
            bool needsDownload = !IsModelDownloaded(model);
            if (needsDownload) {
                DrawDownload(model);
            } else {
                DrawModelInfo(model);
            }

        }

        private void DrawDownload(DiffusionModel model) {
            GUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(model.Name);

            if (GUILayout.Button("Download", GUILayout.Width(70))) {
                DownloadUtils.DownloadModel(model);
            }
            GUILayout.EndHorizontal();
        }

        private void DrawModelInfo(DiffusionModel model) {
            GUILayout.BeginHorizontal();

            EditorGUILayout.LabelField(model.Name, GUILayout.ExpandWidth(true));
            //GUILayout.Button("Serialize to StreamingAssets");

            GUILayout.EndHorizontal();
        }

        private bool IsModelDownloaded(DiffusionModel model) {
            return StableDiffusionPipeline.IsModelAvailable(model);
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