🚧 Experimental! – This package is in an early, exploratory stage and not production-ready. Expect sharp edges, breaking changes, and general chaos.

<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Diffusers
A Unity package to run pretrained diffusion models with Unity Sentis

[OpenUPM] · [Documentation (coming soon)] · [Feedback/Questions]

## About

This is essentially a port of Hugging Face’s [diffusers] library.

As of today only a limited number of pipelines and schedulers are supported (see below).

### Installation

<details>
  <summary> via OpenUPM </summary>
  
 1. In `Edit -> Project Settings -> Package Manager`, add a new scoped registry:

        Name: Doji
        URL: https://package.openupm.com
        Scope(s): com.doji
 
  4. In the Package Manager install `com.doji.diffusers` either by name or select it in the list under `Package Manager -> My Registries`
  5. For the time being,  you also have to use a custom fork of Sentis: In the package Manager -> `Package Manager -> Install package from git URL-> https://github.com/julienkay/com.unity.sentis.git`

</details>

### Features/Roadmap
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
- [ ] more pipeline/model types
  - [x] txt2img
  - [x] img2img,
  - [ ] inpaint,
  - [ ] upscale,
  - [ ] depth2img
  - [ ] marigold
- [x] async pipeline methods (sliced inference)
- [ ] support multiple images per prompt
- [ ] write some documentation

### Dependencies
- [com.doji.transformers]
- [com.doji.pngcs]

[OpenUPM]: https://openupm.com/packages/com.doji.diffusers
[Documentation (coming soon)]: https://github.com/julienkay/com.doji.diffusers
[Feedback/Questions]: https://discussions.unity.com/t/stable-diffusion-diffusers-transformers-package/332701?u=julienkay
[diffusers]: https://github.com/huggingface/diffusers
[com.doji.transformers]: https://github.com/julienkay/com.doji.transformers
[com.doji.pngcs]: https://github.com/julienkay/com.doji.pngcs
