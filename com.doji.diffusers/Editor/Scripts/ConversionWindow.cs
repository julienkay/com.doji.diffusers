using UnityEditor;
using UnityEngine;
using Unity.InferenceEngine;

namespace Doji.AI.Diffusers.Editor {

    public class ConversionWindow : EditorWindow {

        private QuantizationType? _selectedQuantizationType;
        private string ModelId { get; set; }

        public static void ShowWindow(string modelId) {
            ConversionWindow window = GetWindow<ConversionWindow>("Conversion Settings");
            window.minSize = new Vector2(300, 150);
            window.ModelId = modelId;
            window.Show();
        }

        private void OnGUI() {
            EditorGUILayout.LabelField("Configure Conversion", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Model ID", ModelId);
            DrawQuantizationTypeDropdown();
            GUILayout.Space(20);

            if (GUILayout.Button("Convert to .sentis")) {
                PerformConversion();
            }
        }

        // Draw dropdown with a "None" option
        private void DrawQuantizationTypeDropdown() {
            string[] options = new string[] { "None (Float32)", "Float16", "Uint8" };
            int selectedIndex = _selectedQuantizationType.HasValue ? (int)_selectedQuantizationType.Value + 1 : 0;
            int newIndex = EditorGUILayout.Popup("Quantization Type", selectedIndex, options);
            if (newIndex == 0) {
                _selectedQuantizationType = null;
            } else {
                _selectedQuantizationType = (QuantizationType)(newIndex - 1);
            }
        }

        private void PerformConversion() {
            EditorUtility.DisplayProgressBar("Model Hub - Serialize to StreamingAssets", $"Serializing '{ModelId}' to StreamingAssets...", 0.1f);
            ModelHub.ConvertModel(new DiffusionModel(ModelId), _selectedQuantizationType);
            EditorUtility.ClearProgressBar();
        }
    }
}