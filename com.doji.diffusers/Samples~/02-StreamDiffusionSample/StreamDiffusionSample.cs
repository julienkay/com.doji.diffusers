using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using static UnityEngine.Networking.UnityWebRequest;

namespace Doji.AI.Diffusers.Samples {

    public class StreamDiffusionSample : MonoBehaviour {

        StreamDiffusion stream;
        Tensor<float> init_image;

        private const int WIDTH = 512;
        private const int HEIGHT = 512;
        public RenderTexture Result;

        private void Start() {
            // load any model using StableDiffusionPipeline
            var pipe = StableDiffusionPipeline.FromPretrained("julienkay/sd-turbo");

            // Wrap the pipeline in StreamDiffusion
            stream = new StreamDiffusion(
                pipe,
                tIndexList: new List<int>() { 32, 45 }
            );

            string prompt = "1girl with dog hair, thick frame glasses";
            // Prepare the stream
            stream.Prepare(prompt);

            // Prepare image
            //TensorFloat init_image = load_image("assets/img2img_example.png").resize((512, 512))

            // Warmup >= len(t_index_list) x frame_buffer_size
            stream.Update(null);
            stream.Update(null);

            if (Result == null) {
                Result = new RenderTexture(WIDTH, HEIGHT, 0, RenderTextureFormat.ARGB32);
            }   
        }

        private void Update() {
            // Run the stream infinitely
            var x_output = stream.Update(init_image);
            TextureConverter.RenderToTexture(x_output, Result);
        }
    }
}