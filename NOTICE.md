# Third-Party Notices

This pipeline depends on third-party models and software. Operators must comply with
each component's license at deployment time. Standard MIT/Apache/BSD Python and Node
dependencies are covered by their own `LICENSE` files inside the installed packages and
`node_modules/`; this file lists the components that carry attribution, gating, or
non-permissive redistribution terms.

The pipeline makes **no closed-source API calls** during generation. Every model runs
locally inside the container on the operator's own hardware.

---

## Qwen-3.6-27B-SwiftHeron — coder (scene program generation)

- **Component:** `cont1037/Qwen-3.6-27B-SwiftHeron` (private mirror).
- **Upstream:** `computer-vision-ai-lab/Qwen-3.6-27B-SwiftHeron`
  @ `73d641576ca5c881bc383eef27f5c374ac0d41d6`, itself derived from Alibaba's Qwen family.
- **Used in:** `pipeline_service/modules/scene_coder/` — turns the reference image into a
  Three.js ES module. Served by SGLang on GPU 0.
- **Form:** FP8 (`compressed-tensors`, W8A8, per-channel weights / per-token dynamic
  activations). The vision tower is excluded from quantization and stays bf16.
- **License:** the upstream Qwen license (Apache-2.0 family, commercial use permitted).
  The mirror redistributes the upstream weights unmodified; operators should read the
  upstream model card before redeploying.

## Qwen3.6-27B-DFlash-SwiftHeron — speculative drafter

- **Component:** `cont1037/Qwen3.6-27B-DFlash-SwiftHeron` (private).
- **Used in:** SGLang speculative decoding (`--speculative-algorithm DFLASH`) for the
  coder endpoint. A `DFlashDraftModel` block-fill drafter (5 layers) trained against the
  64-layer coder target (`target_layer_ids: [1, 16, 31, 46, 61]`).
- **License:** inherits the base Qwen license, as a derivative of the coder model.

## GLM-4.6V-Flash (Zhipu AI / Z.ai) — VLM judge and critic

- **Component:** `zai-org/GLM-4.6V-Flash` @ `411bb4d77144a3f03accbf4b780f5acb8b7cde4e`.
- **Used in:** `pipeline_service/modules/judge/multi_stage.py` — every multi-stage judge
  VLM call. Served by vLLM across GPUs 1-3 (`data_parallel_size: 3`).
- **Provider:** Zhipu AI / Z.ai.
- **License:** GLM-4 / GLM-4V License — see
  <https://huggingface.co/zai-org/GLM-4.6V-Flash>. **This repository does not
  redistribute the weights**; the container pulls them from the upstream repository at
  runtime, deliberately left un-mirrored so the upstream license and model card remain
  the operator's point of reference.
- **Citation:** Zhipu AI / Z.ai, *GLM-4.6V-Flash*. <https://github.com/zai-org/GLM-4.6V>

## DINOv3 (Meta) — judge best-view selection

- **Component:** `cont1037/dinov3-vits16-pretrain-lvd1689m`
  @ `ba4586919549aa692fd9b39ab3e9777c564b497d`.
- **Upstream:** derived from Meta's `facebook/dinov3-vits16-pretrain-lvd1689m`.
- **Used in:** `pipeline_service/modules/judge/dino.py` — per-view embeddings feeding the
  judge's Stage 2A best-view comparison, and the bracket's draw tie-break.
- **Provider:** Meta AI Research.
- **License:** DINOv3 License (custom Meta license). The upstream model is **gated** on
  Hugging Face and its license requires attribution and restricts certain uses
  (medical / safety-critical contexts, among others).
  > **Operator note:** this mirror redistributes weights derived from a gated,
  > custom-licensed Meta model. Operators are responsible for confirming that their
  > redistribution and use comply with Meta's DINOv3 license terms; accepting those
  > terms at
  > <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>
  > is the safest path.
- **Citation:**
  > Oriane Siméoni et al., *DINOv3*, Meta AI Research, 2024.
  > <https://github.com/facebookresearch/dinov3> · <https://arxiv.org/abs/2508.10104>

## Three.js — scene construction and rendering

- **Component:** `three` (pinned in `docker/package.json` / `package-lock.json`).
- **Used in:** the renderer sidecar (`pipeline_service/modules/renderer/render_service/`)
  and inside the validator sandbox.
- **License:** MIT. <https://github.com/mrdoob/three.js>

## Inference servers

- **SGLang** — serves the coder with DFlash speculative decoding. Built into the image by
  `docker/setup_sglang_env.sh`, pinned by commit and constrained by
  `docker/sglang-env.lock.txt`. Apache-2.0. <https://github.com/sgl-project/sglang>
- **vLLM** — serves GLM-4.6V-Flash. Built by
  `pipeline_service/scripts/setup_glm_vllm_env.sh`. Apache-2.0.
  <https://github.com/vllm-project/vllm>
- **Puppeteer / Chromium** — headless rendering of candidate scenes. Apache-2.0 / BSD.

## Judge implementation

The multi-stage duel judge in `pipeline_service/modules/judge/multi_stage.py` is a port
of the 404-GEN subnet's own judge (`judge-service/judge_service/judges/multi_stage.py`),
adapted to consume locally rendered views in memory instead of fetching them from a CDN.
Prompt strings, thresholds, and stage logic are kept identical so local candidate
selection matches the scoring the submission will actually face. The subnet codebase is
MIT-licensed.
