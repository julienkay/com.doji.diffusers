using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Doji.AI.Diffusers.Samples {

    public class StableDiffusionImg2ImgUI : MonoBehaviour {

        public StableDiffusionUI UI { get; private set; }

        public TMP_InputField PromptField;
        public Button GenerateButton;
        public Texture2D InputImage;
        public RawImage Input;

        public float Strength = 0.8f;

        private void Awake() {
            UI = GetComponentInParent<StableDiffusionUI>();
            GenerateButton.onClick.AddListener(OnGenerateClicked);
            Input.texture = InputImage;
            UI.InputImage = InputImage;
        }

        private void OnGenerateClicked() {
            UI.Img2Img(PromptField.text, Strength);
        }

#if UNITY_EDITOR
        private void OnValidate() {
            if (Input.texture != Input) {
                Input.texture = InputImage;
                if (UI != null) {
                    UI.InputImage = InputImage;
                }
            }
        }
#endif
    }
}