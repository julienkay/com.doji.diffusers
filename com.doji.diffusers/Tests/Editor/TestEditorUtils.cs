using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Doji.AI.Diffusers.Editor.Tests {

    public class TestEditorUtils {
        [MenuItem("Window/Diffusers/Latents Generator")]
        public static void Generate() {
            var window = EditorWindow.GetWindow(typeof(LatentsGenerator), false, "Latents Generator");
        }
    }

    public class LatentsGenerator : EditorWindow {

        [SerializeField]
        private int _width = 64;
        [SerializeField]
        private int _height = 64;

        [SerializeField]
        private int _batchSize = 1;
        [SerializeField]
        private int _numImagesPerPrompt = 1;

        private float[] _latents;
        [SerializeField]
        private Texture2D _latentImage;

        private SerializedObject _serializedObject;
        private SerializedProperty _widthProp;
        private SerializedProperty _heightProp;
        private SerializedProperty _batchSizeProp;
        private SerializedProperty _numImagesProp;
        private SerializedProperty _latentImageProp;

        private void OnEnable() {
            _serializedObject = new SerializedObject(this);

            _widthProp = _serializedObject.FindProperty("_width");
            _heightProp = _serializedObject.FindProperty("_height");
            _batchSizeProp = _serializedObject.FindProperty("_batchSize");
            _numImagesProp = _serializedObject.FindProperty("_numImagesPerPrompt");
            _latentImageProp = _serializedObject.FindProperty("_latentImage");

            GenerateLatents();
        }

        private void OnGUI() {
            _serializedObject.Update();

            EditorGUILayout.DelayedIntField(_widthProp, new GUIContent("Width"));
            if (_widthProp.intValue <= 0) {
                _widthProp.intValue = 64;
            }
            EditorGUILayout.DelayedIntField(_heightProp, new GUIContent("Height"));
            if (_heightProp.intValue <= 0) {
                _heightProp.intValue = 64;
            }

            //int n = EditorGUILayout.IntField("Num Images", _numImagesProp.intValue);
            //_numImagesPerPrompt = Mathf.Clamp(n, 1, 4);
            //int b = EditorGUILayout.IntField("Batch Size", _batchSizeProp.intValue);
            //_batchSize = Mathf.Clamp(b, 1, 4);


            if (_latentImage != null) {
                _latentImageProp.objectReferenceValue = (Texture2D)EditorGUILayout.ObjectField("", (Texture2D)_latentImageProp.objectReferenceValue, typeof(Texture2D), false, GUILayout.ExpandWidth(false));

            }

            GUILayout.BeginHorizontal();

            SaveTxt();
            SaveBinary();
            SaveImage();

            GUILayout.EndHorizontal();

            if (_serializedObject.ApplyModifiedProperties()) {
                // generate latents if properties changed
                GenerateLatents();
                Repaint();
            }
        }

        private void SaveTxt() {
            if (GUILayout.Button("Save .txt")) {
                if (_latents == null) {
                    return;
                }
                string text = string.Join(", ", _latents);
                string path = EditorUtility.SaveFilePanel("Save latents as .txt", "", GetFileName(), ".txt");
                if (string.IsNullOrEmpty(path) || !Directory.Exists(Path.GetDirectoryName(path))) {
                    return;
                }
                File.WriteAllText(path, text);
            }
        }

        private void SaveBinary() {
            if (GUILayout.Button("Save .bin")) {
                if (_latents == null) {
                    return;
                }
                var byteArray = new byte[_latents.Length * 4];
                Buffer.BlockCopy(_latents, 0, byteArray, 0, byteArray.Length);

                string path = EditorUtility.SaveFilePanel("Save latents as .bin", "", GetFileName(), ".bin");
                if (string.IsNullOrEmpty(path) || !Directory.Exists(Path.GetDirectoryName(path))) {
                    return;
                }
                File.WriteAllBytes(path, byteArray);
            }
        }

        private void SaveImage() {
            if (GUILayout.Button("Save .png")) {
                if (_latentImage == null) {
                    return;
                }
                byte[] pngData = _latentImage.EncodeToPNG();
                string path = EditorUtility.SaveFilePanel("Save latents as .png", "", GetFileName(), ".png");
                if (string.IsNullOrEmpty(path) || !Directory.Exists(Path.GetDirectoryName(path))) {
                    return;
                }
                File.WriteAllBytes(path, pngData);
            }
        }

        private string GetFileName() {
            return $"latents_{_latentImage.name}_1_4_{_height}_{_width}";
        }

        private void GenerateLatents() {
            int width = _width;
            int height = _height;
            int size = _batchSize * _numImagesPerPrompt * 4 * height * width;
            int seed = new System.Random().Next();
            _latents = ArrayUtils.Randn(size, 0, 1, new System.Random(seed));
            if (_latentImage != null) {
                DestroyImmediate(_latentImage);
            }

            _latentImage = new Texture2D(width, height, TextureFormat.ARGB32, false, true);
            _latentImage.name = seed.ToString();
            var data = _latentImage.GetPixelData<Color32>(0);
            int index = 0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    Color32 c = new Color32(
                        (byte)((_latents[index + 0] / 2f + 0.5f) * 255f),
                        (byte)((_latents[index + 1] / 2f + 0.5f) * 255f),
                        (byte)((_latents[index + 2] / 2f + 0.5f) * 255f),
                        (byte)((_latents[index + 3] / 2f + 0.5f) * 255f)
                    );
                    data[i * width + j] = c;
                    index += 4;
                }
            }
            _latentImage.SetPixelData(data, 0);
            _latentImage.Apply();
        }
    }
}