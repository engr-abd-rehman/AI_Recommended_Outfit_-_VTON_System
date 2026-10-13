# Third-Party Sources and Asset Terms

## OOTDiffusion

- Official source: https://github.com/levihsu/OOTDiffusion
- Pinned commit: `13ef0faba266cdde9febc8ad39be2395bbb89d9c`.
- Authors: Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen.
- Upstream source license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
- License copy: [licenses/OOTDiffusion-CC-BY-NC-SA-4.0.txt](licenses/OOTDiffusion-CC-BY-NC-SA-4.0.txt).
- This repository distributes only a small local patch; the full upstream checkout
  is fetched during setup and remains ignored. The patch removes unused debug/config
  imports in OpenPose. The existing local inference-script whitespace change is
  behavior-neutral and is not needed for reproduction.

The patch is an adaptation of upstream source and follows its upstream license;
it is not relicensed as project-owned permissive code. The downloaded tree also
contains components with their own notices; preserve them. The integration does
not imply endorsement by the model authors or permission for commercial use.

## Model Files

OOTDiffusion models are downloaded directly from
[levihsu/OOTDiffusion](https://huggingface.co/levihsu/OOTDiffusion). The separate
CLIP image encoder comes from
[openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14).
Their pinned revisions and destinations are documented in [setup](SETUP_GUIDE.md).
No model weight, parsing ONNX, pose PTH, or duplicate checkpoint copy is published.
Review the relevant model cards and any underlying model/dataset terms before use.

Recommendation embeddings use the locally installed
[OpenAI CLIP implementation](https://github.com/openai/CLIP). Installation through
`openai-clip` does not require an API key or grant ownership of pretrained models.

## Dataset

The processed archive is linked by the
[Text2Human data preparation guide](https://github.com/yumingj/Text2Human#1-dataset-preparation).
Its underlying data is [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal).
The [provider's agreement](https://github.com/yumingj/DeepFashion-MultiModal#agreement)
restricts research use and further distribution. This release conservatively
excludes the dataset, copied annotation metadata, derived vector index, and
screenshots containing catalog pixels. Obtain data from the original provider
and build the catalog locally after reviewing those terms.

## Photos and Screenshots

Existing local stock images, model photos, uploaded inputs, catalog crops, and
generated result images are not included. Their provenance and redistribution
rights have not been established by this audit. The publication screenshots in
`docs/images/` contain only the source-only interface with generic local-demo
identity and no model photograph or catalog thumbnails.

For new examples, use assets you created or are explicitly permitted to process
and distribute. A successful inference run is not an asset-rights clearance.

## Interface Dependencies

Flask/Jinja, PyTorch, NumPy, Pillow, OpenCV, Diffusers, Transformers, and other
packages retain their own licenses. Package pins are installation instructions,
not a redistribution of those packages. Bootstrap and Font Awesome styles are
loaded from the CDN URLs in `templates/index.html`; their original notices and
terms apply. These libraries do not change the model/dataset restrictions.

## References

- Xu et al., *OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on*, [paper](https://arxiv.org/abs/2403.01779).
- Jiang et al., *Text2Human: Text-Driven Controllable Human Image Generation*, SIGGRAPH 2022, [project](https://yumingj.github.io/projects/Text2Human.html).

These notes identify sources and the release's exclusion policy. They are not a
legal opinion or a guarantee that any particular deployment is authorized.
