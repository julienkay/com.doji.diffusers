<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Diffusers
A Unity package to run pretrained diffusion models with Unity Sentis

[OpenUPM] · [Documentation (coming soon)] · [Feedback/Questions]

## About

This is essentially a port of Hugging Face’s [diffusers] library.

It is still ***very*** early though, so as of today we only support:
- a simple Stable Diffusion pipeline compatible with Stable Diffusion 1.5
- the default PNDMScheduler

### Roadmap
Some things that might be worked on next are:
- add/fix classifier-free guidance
- async pipeline methods
- pipelines for img2img, inpaint, upscale, depth2img
- support for other models (2.1, SDXL, sdxl-turbo, LoRA models)
- more scheduler implementations
- support multiple images per prompt
- write some documentation

[OpenUPM]: https://openupm.com/packages/com.doji.diffusers
[Documentation (coming soon)]: https://github.com/julienkay/com.doji.diffusers
[Feedback/Questions]: https://discussions.unity.com/t/stable-diffusion-diffusers-transformers-package/332701
[diffusers]: https://github.com/huggingface/diffusers
