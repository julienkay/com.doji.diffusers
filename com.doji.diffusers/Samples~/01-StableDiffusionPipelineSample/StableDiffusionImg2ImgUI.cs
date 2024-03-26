using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Doji.AI.Diffusers.Samples {

    public class StableDiffusionImg2ImgUI : MonoBehaviour {

        public StableDiffusionUI UI { get; private set; }

        public TMP_InputField PromptField;
        public Button GenerateButton;
        public Texture2D Input;
        public RawImage InputImage;
        public RawImage ResultImage;

        public float Strength = 0.8f;

        private void Awake() {
            UI = GetComponentInParent<StableDiffusionUI>();
            GenerateButton.onClick.AddListener(OnGenerateClicked);
            InputImage.texture = Input;
            UI.InputImage = Input;
        }

        private void OnGenerateClicked() {
            UI.Img2Img(PromptField.text, Strength);
            ResultImage.texture = UI.Result;
        }

#if UNITY_EDITOR
        private void OnValidate() {
            if (InputImage.texture != InputImage) {
                InputImage.texture = Input;
                if (UI != null) {
                    UI.InputImage = Input;
                }
            }
        }
#endif
    }
}