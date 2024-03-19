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

        private void Awake() {
            UI = GetComponentInParent<StableDiffusionUI>();
            PromptField.onValueChanged.AddListener(OnPromptChanged);
            GenerateButton.onClick.AddListener(OnGenerateClicked);
            Input.texture = InputImage;
            UI.InputImage = InputImage;
        }

        private void OnGenerateClicked() {
            UI.Img2Img();
        }

        private void OnPromptChanged(string value) {
            UI.Prompt = value;
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