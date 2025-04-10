using UnityEditor;

namespace Doji.AI.Diffusers.Samples.Editor {
    [CustomEditor(typeof(StableDiffusionUI))]
    public class StableDiffusionUIEditor : UnityEditor.Editor {
        SerializedProperty modelDropdownProp;
        SerializedProperty modelProp;
        SerializedProperty customModelProp;
        SerializedProperty resolutionProp;
        SerializedProperty stepsProp;
        SerializedProperty guidanceScaleProp;
        SerializedProperty inputImageProp;
        SerializedProperty resultProp;

        void OnEnable() {
            modelDropdownProp = serializedObject.FindProperty("ModelDropdown");
            modelProp = serializedObject.FindProperty("Model");
            customModelProp = serializedObject.FindProperty("CustomModel");
            resolutionProp = serializedObject.FindProperty("Resolution");
            stepsProp = serializedObject.FindProperty("Steps");
            guidanceScaleProp = serializedObject.FindProperty("GuidanceScale");
            inputImageProp = serializedObject.FindProperty("InputImage");
            resultProp = serializedObject.FindProperty("Result");
        }

        public override void OnInspectorGUI() {
            serializedObject.Update();

            EditorGUILayout.PropertyField(modelDropdownProp);
            EditorGUILayout.PropertyField(modelProp);

            if (modelProp.enumValueIndex == (int)Model.Custom) {
                EditorGUILayout.PropertyField(customModelProp);
            }

            EditorGUILayout.PropertyField(resolutionProp);
            EditorGUILayout.PropertyField(stepsProp);
            EditorGUILayout.PropertyField(guidanceScaleProp);
            EditorGUILayout.PropertyField(inputImageProp);
            EditorGUILayout.PropertyField(resultProp);

            serializedObject.ApplyModifiedProperties();
        }
    }
}
