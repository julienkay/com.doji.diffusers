using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Doji.AI.Diffusers.Samples {

    public class StableDiffusionTxt2ImgUI : MonoBehaviour {

        public StableDiffusionUI UI { get; private set; }

        public TMP_InputField PromptField;
        public Button GenerateButton;

        private void Awake() {
            UI = GetComponentInParent<StableDiffusionUI>();
            PromptField.onValueChanged.AddListener(OnPromptChanged);
            GenerateButton.onClick.AddListener(OnGenerateClicked);
        }

        private async void OnGenerateClicked() {
            UI.Txt2Img();
            /*try {
                await UI.Txt2ImgAsync();
            } catch (Exception ex) {
                throw ex;
            }*/
        }

        private void OnPromptChanged(string value) {
            UI.Prompt = value;
        }

    }
}