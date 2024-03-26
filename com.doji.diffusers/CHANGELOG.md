# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2024-03-26

### Fixed

- Use a Resources folder inside the project folder instead, to make model download work when package is immutable

## [0.3.0] - 2024-03-26

### Added

- Support for Turbo models (SDXL Turbo & SD Turbo)
- An EulerAncestralDiscreteScheduler implementation
- Async pipeline methods
- Convert diffusion pipeline models from .onnx to .sentis and load from StreamingAssets
- Image-to-image pipelines for existing models

## [0.2.0] - 2024-03-02

### Added

- Support for Stable Diffusion XL
- An EulerDiscreteScheduler implementation
- 'seed' parameter for deterministic generation
- Saving parameters (prompt, seed, model, etc.) as PNG metadata

## [0.1.0] - 2024-02-16

### Added

- Support for Stable Diffusion 2.1
- A DDIMScheduler implementation
- Support for negative prompts

### Fixed

- Fix Classifier-Free Guidance not working correctly

## [0.0.0] - 2024-01-29

- Initial Release
- Supports Stable Diffusion 1.5