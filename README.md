<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Diffusers
A Unity package to run pretrained diffusion models with Unity Sentis

[OpenUPM] · [Documentation (coming soon)] · [Feedback/Questions]

## About

This is essentially a port of Hugging Face’s [diffusers] library.

It is still ***very*** early though, so as of today only a limited number of pipelines and schedulers are supported (see below).

### Roadmap
Some things that might be worked on next are:
- [x] add/fix classifier-free guidance
- [ ] support more models
  - [x] SD 1.5
  - [x] SD 2.1
  - [x] SDXL
  - [x] SD-Turbo
  - [x] SDXL-Turbo
  - [ ] LoRA models
- [ ] more scheduler implementations
  - [x] PNDM
  - [x] DDIM
  - [x] EulerDiscrete
  - [x] EulerAncestralDiscrete
  - [ ] LCM
  - [ ] DDPM
  - [ ] KDPM2Discrete
  - [ ] KDPM2AncestralDiscrete
- [ ] async pipeline methods (sliced inference)
- [ ] pipelines for img2img, inpaint, upscale, depth2img
- [ ] support multiple images per prompt
- [ ] write some documentation

### Dependencies
- [com.doji.transformers]
- [com.doji.pngcs]

[OpenUPM]: https://openupm.com/packages/com.doji.diffusers
[Documentation (coming soon)]: https://github.com/julienkay/com.doji.diffusers
[Feedback/Questions]: https://discussions.unity.com/t/stable-diffusion-diffusers-transformers-package/332701
[diffusers]: https://github.com/huggingface/diffusers
[com.doji.transformers]: https://github.com/julienkay/com.doji.transformers
[com.doji.pngcs]: https://github.com/julienkay/com.doji.pngcs