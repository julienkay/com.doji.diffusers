using Doji.AI.Diffusers;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

/// <summary>
/// Custom Texture Importer Editor that displays generation parameters
/// for pngs that were generated using this package.
/// </summary>
/// <remarks>
/// https://forum.unity.com/threads/how-to-add-custom-editor-to-assetpostprocessor.1161476/
/// </remarks>
[CustomEditor(typeof(TextureImporter))]
public class TextureImporterExtension : Editor {

    AssetImporterEditor _defaultEditor;

    void OnEnable() {
        if (_defaultEditor == null) {
            _defaultEditor = (AssetImporterEditor)AssetImporterEditor.CreateEditor(targets,
                Type.GetType("UnityEditor.TextureImporterInspector, UnityEditor"));
            MethodInfo dynMethod = Type.GetType("UnityEditor.TextureImporterInspector, UnityEditor")
                .GetMethod("InternalSetAssetImporterTargetEditor", BindingFlags.NonPublic | BindingFlags.Instance);
            dynMethod.Invoke(_defaultEditor, new object[] { this });
        }
    }

    void OnDisable() {
        _defaultEditor.OnDisable();
    }

    void OnDestroy() {
        _defaultEditor.OnEnable();
        DestroyImmediate(_defaultEditor);
    }

    public override void OnInspectorGUI() {
        _defaultEditor.OnInspectorGUI();

        GUILayout.Space(20);
        string assetPath = (_defaultEditor.target as TextureImporter).assetPath;
       
        if (GetMetadata(assetPath, out Parameters p)) {
            GUILayout.Label("Generation Parameters", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            //EditorGUI.BeginDisabledGroup(true);
            ReadOnlyTextField("Model", p.Model);
            ReadOnlyTextField("Pipeline", p.Pipeline);
            ReadOnlyTextArea("Prompt", p.Prompt);
            ReadOnlyTextArea("NegativePrompt", p.NegativePrompt);
            ReadOnlyTextField("Steps", p.Steps.ToString());
            ReadOnlyTextField("Sampler", p.Sampler);
            ReadOnlyTextField("CfgScale", p.CfgScale.ToString());
            ReadOnlyTextField("Seed", p.Seed == null ? "None" : p.Seed?.ToString());
            ReadOnlyTextField("Width", p.Width.ToString());
            ReadOnlyTextField("Height", p.Height.ToString());
            ReadOnlyTextField("Eta", p.Eta == null ? "None" : p.Eta.ToString());

            //EditorGUI.EndDisabledGroup();
            EditorGUI.indentLevel--;
        }

        serializedObject.ApplyModifiedProperties();

        // Unfortunately we cant hide the 'ImportedObject' section
        // this just moves it out of view
        //GUILayout.Space(2048);
    }

    private void ReadOnlyTextField(string label, string text) {
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField(label, GUILayout.Width(EditorGUIUtility.labelWidth - 4));
        EditorGUILayout.SelectableLabel(text, EditorStyles.textField, GUILayout.Height(EditorGUIUtility.singleLineHeight));
        EditorGUILayout.EndHorizontal();
    }

    private void ReadOnlyTextArea(string label, string text) {
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField(label, GUILayout.Width(EditorGUIUtility.labelWidth - 4));
        EditorGUILayout.SelectableLabel(text, EditorStyles.textArea);
        EditorGUILayout.EndHorizontal();
    }

    /// <summary>
    /// Returns generation metadata for PNG images created with this package.
    /// Returns false for other file types or PNGs that do not contain
    /// the expected metadata key.
    /// </summary>
    private bool GetMetadata(string filePath, out Parameters p) {
        p = null;
        if (!Path.GetExtension(filePath).Equals(".png", StringComparison.OrdinalIgnoreCase)) {
            return false;
        }

        string metadata = PNGUtils.GetMetadata(filePath);
        if (string.IsNullOrEmpty(metadata)) {
            return false;
        }

        p = JsonConvert.DeserializeObject<Parameters>(metadata);
        if (p == null) {
            return false;
        }

        return true;
    }
}