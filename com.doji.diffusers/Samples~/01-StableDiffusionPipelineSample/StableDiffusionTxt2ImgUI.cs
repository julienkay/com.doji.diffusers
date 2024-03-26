using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Doji.AI.Diffusers.Samples {

    public class StableDiffusionTxt2ImgUI : MonoBehaviour {

        public StableDiffusionUI UI { get; private set; }

        public TMP_InputField PromptField;
        public Button GenerateButton;
        public RawImage ResultImage;

        private void Awake() {
            UI = GetComponentInParent<StableDiffusionUI>();
            GenerateButton.onClick.AddListener(OnGenerateClicked);
        }

        private async void OnGenerateClicked() {
            UI.Txt2Img(PromptField.text);
            /*try {
                await UI.Txt2ImgAsync(PromptField.text);
            } catch (Exception ex) {
                throw ex;
            }*/
            ResultImage.texture = UI.Result;
        }
    }
}