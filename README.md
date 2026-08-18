<!-- EKTOME FORK NOTICE -->
# Zynerji/vllm — fork notice

**A fork of [vllm-project/vllm](https://github.com/vllm-project/vllm) carrying three patches that
let vLLM serve checkpoints whose `embed_tokens` and `lm_head` are quantized.**

Upstream quantizes the transformer blocks happily but assumes the two vocabulary projections stay
in full precision. On Qwen3.8-27B those are two `248320 x 5120` tensors — roughly 6 GB in BF16 —
which is the difference between "fits on a 24 GB card with long context" and "does not fit".
Everything here exists to close that gap. Nothing changes sampling, scheduling, or numerics for an
ordinary checkpoint.

Built for [`Zynerji/Qwen3.8-27B-PristinelyUncensored-HOMEUSER-16-24`](https://huggingface.co/Zynerji/Qwen3.8-27B-PristinelyUncensored-HOMEUSER-16-24),
but none of it is model-specific beyond the Qwen3.5 file paths.

## Install

```bash
pip install git+https://github.com/Zynerji/vllm@ektome-homeuser-v2
```

> **The patches live on `ektome-homeuser-v2`, not `main`.** `main` tracks upstream unchanged —
> cloning the default branch gets you stock vLLM.

## What changed, and why

Three commits, four files, **+81 lines and 0 deletions** on top of upstream `main`.

### 1. Pass `quant_config` and `prefix` to Qwen3.5 `embed_tokens`
`vllm/model_executor/models/qwen3_5.py`, `qwen3_5_mtp.py` · +7/+7

Both `Qwen3_5Model` and `Qwen3_5MultiTokenPredictor` built their `VocabParallelEmbedding` without
`quant_config` or `prefix`. Against a compressed-tensors checkpoint that quantizes the embedding
this silently constructs an **unquantized** embedding, and loading dies with:

```
no module or parameter named 'embed_tokens.weight_packed'
```

`prefix` is load-bearing, not cosmetic: compressed-tensors matches schemes **by layer name**, so
without it the target regex cannot resolve. `llama.py` already passes both — this brings the Qwen
paths in line. The MTP site matters separately: without it the speculative head fails at engine
init with the same error inside `Qwen3_5MultiTokenPredictor`.

### 2. Expose `LinearBase` geometry on `ParallelLMHead`
`vllm/model_executor/layers/vocab_parallel_embedding.py` · +13

`ParallelLMHead` is the only quantizable projection in vLLM that does **not** derive from
`LinearBase`. Quantization schemes read geometry off the module they are handed, so a checkpoint
that quantizes `lm_head` makes schemes such as `CompressedTensorsW8A16Fp8` raise `AttributeError`.

The patch sets the six attributes they look for — `output_partition_sizes`, `logical_widths`,
`output_size_per_partition`, `input_size`, `input_size_per_partition`, `has_bias` — each derived
from state the layer already tracks. No behaviour change for an unquantized `lm_head`.

### 3. Auto-enable MTP when the checkpoint ships a draft head
`vllm/config/vllm.py` · +64

`speculative_config` defaults to `None`, so a checkpoint carrying a multi-token-prediction head
gets no benefit from it unless the caller knows to ask — and nothing in a model repo can signal
that, because it is an engine argument rather than model metadata.

`VllmConfig._maybe_auto_enable_mtp()` runs from `__post_init__`, looks for
`num_nextn_predict_layers` / `mtp_num_hidden_layers` on the HF config, and builds a
`SpeculativeConfig(method="mtp")` when it finds one. Speculative decoding is
**distribution-preserving** — drafts are verified against the target model — so this cannot change
outputs, only speed. It is wrapped so that a failure to construct the config logs a warning and
continues rather than blocking startup.

| variable | default | effect |
|---|---|---|
| `VLLM_AUTO_MTP` | `1` | set `0` to opt out entirely |
| `VLLM_AUTO_MTP_TOKENS` | `1` | `num_speculative_tokens` |

**The payoff is hardware-dependent and not always positive.** Measured on this checkpoint family:
**1.49× on a 3090 Ti, 0.92× on a 5090** — Ampere gains, Ada and Blackwell lose. It will not
initialise at all on a 16 GB card, which has no headroom for the draft model. Treat no single
number as a family figure; if it costs you throughput, set `VLLM_AUTO_MTP=0`.

## Upstreaming

All three are narrow: two are consistency fixes bringing Qwen3.5 and `ParallelLMHead` in line with
what other models and layers already do, and the third is opt-out-able and cannot change outputs.
No private APIs, no vendored dependencies, no new build steps — rebasing onto a newer upstream
should be mechanical.

## A note on `ektome-homeuser` (the previous branch)

The original branch is retained so existing pins keep resolving, but **it should not be used**. Its
three commits were made with CRLF line endings against LF originals, so git recorded every file as
a whole-file rewrite: an 11-line addition showed as `@@ -1,581 +1,594 @@`, and `vllm/config/vllm.py`
reported `+2737/-2679` for 58 net lines. That made the diff unreviewable, destroyed `git blame`,
guaranteed a conflict on every line of any rebase, and left the branch unupstreamable. It was also
missing the `qwen3_5_mtp.py` half of patch 1 while *enabling* MTP by default — a combination that
fails at engine init on exactly the checkpoints this fork exists to serve.

`ektome-homeuser-v2` is that work rebuilt cleanly on current upstream: LF throughout, pure
additions, and the missing patch included.

---
<!-- END EKTOME FORK NOTICE -->

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
