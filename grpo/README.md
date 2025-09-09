# CS 674 GRPO Lab 

## 1 Overview

This lab walks you through building an RL pipeline for math problem solving using PPO first (one rollout per prompt) and then extending it to GRPO (multiple rollouts per prompt with group-normalized rewards). The pipeline runs as two cooperating processes:

- Inference worker (gen_worker): runs on a dedicated GPU (configurable via `GEN_DEVICE`) using vLLM to generate model outputs. It computes rewards for each completion and the on-policy per-token log-probabilities (gen_logps), then uploads a batch to the reference server.
- Reference server (ref_server): computes per-token reference log-probabilities (refs) for the completion segment using a frozen model and returns packed batches.
- Trainer (DeepSpeed across remaining GPUs): pulls batches from the reference server, computes the PPO loss with a reference KL term, backprops, and updates the policy.

## 2 Files in this lab

- `grpo_lab.py`: main lab script. Contains TODOs for the core student work.
- `ref_server.py`: reference log-prob server. Provided and ready to use.

## 3 What you will implement (PPO-first, then GRPO)

1) `reward_format(answer: str) -> float`
- Return +1.25 if the output strictly matches `<think>...</think><answer>...</answer>` (exactly one pair of each tag), else -1.0.

2) `reward_correct(item: dict, answer: str) -> float`
- Extract the final numeric answer (int/float/fraction) from the model output and compare to `item["A"]`. Return +1.0 if correct, else -1.0.

3) `gen_worker(...)` (Core PPO inference)
- Use vLLM to generate a single rollout per prompt (PPO uses one rollout).
- Compute rewards (format + correctness) for that rollout.
- Compute on-policy gen_logps for the completion tokens by passing the concatenated `[prompt || completion]` token ids back as `prompt_token_ids` with `prompt_logprobs=1`.
- Package `[plen, merged_ids, rewards, gen_logps]` and POST to the reference server `/upload`.
- Note: In the scaffold, helpers are provided so you only need to implement the main loop.

4) `PPO_step(batch, ...)` (Loss computation)
- Compute model per-token log-probs on `inputs` and slice to the completion region `[plen-1:]`.
- Compute ratio with on-policy gen_logps, apply clipping: `ratio = exp(curr_logps - gen_logps)`.
- Add forward-KL to the reference: `kl = exp(refs - curr_logps) - (refs - curr_logps) - 1`.
- Treat reward as advantage, broadcast per token, mask padding, average over valid completion tokens, then mean over batch.
- Return the scalar loss: negative clipped objective minus `beta * kl`.

At this point, you have PPO RLVR.

5) `normalize_group_rewards(rewards: torch.Tensor) -> torch.Tensor` (Turn PPO → GRPO)
- Add a `GROUP_SIZE` config (start with `GROUP_SIZE = 2 or 3`).
- Update the inference path to generate multiple rollouts per prompt (vLLM `SamplingParams n=GROUP_SIZE`) and collect `GROUP_SIZE` answers and rewards per prompt.
- Normalize rewards within each group: `(rewards - mean) / (std + 1e-4)` before uploading to the server.
- Adjust batching and JSON headers accordingly.

## 4 Environment and setup

- Python 3.10+
- Default model: "Qwen/Qwen2.5-3B".
- Install dependencies (see project `requirements.txt`). Ensure CUDA-compatible PyTorch.
- vLLM requires GPUs; confirm the GPU allocation policy on your cluster.

## 5 How to run

1) Start the reference server (login or compute node with GPU):

```bash
python RLVR_Goal_Directedness/ref_server.py
```

2) Start the trainer (this spawns the generation worker on the `GEN_DEVICE` GPU set in `grpo_lab.py`):

```bash
deepspeed --num_gpus <N> RLVR_Goal_Directedness/grpo_lab.py --train
```

- The generation worker runs inference via vLLM and uploads batches to the reference server.
- The trainer fetches batches from the server and runs PPO updates across the remaining GPUs.


## 6 Deliverables

- Turn in your working implementation of Steps 1–5, and logs showing:
  - A few sampled prompts and generated answers
  - That training has progressed until format adherence rate > 0.8 
  - Average reward increasing over steps; you should be able to produce plots like the following:
    
![Qwen2dot5-7B-res](https://github.com/user-attachments/assets/dcbf3956-e951-4183-9a58-7b932d5ba48d)

## 7 Tips and expectations

- Start with PPO (single rollout per prompt). Keep the loop simple and ensure the payload schema matches the reference server expectations.
- Masking is critical: only average loss over completion tokens (ignore prompt tokens and padding).
- When you extend to GRPO, only then add `GROUP_SIZE` and group normalization logic.
- Debug with small models and short max_tokens first; scale up after the end-to-end path works.

## 8 Troubleshooting

- CUDA OOM: reduce `max_tokens`, batch sizes, and model size; ensure only the trainer GPUs hold optimizer states.
- vLLM errors: verify `MODEL_PATH`, GPU visibility, and driver/runtime versions.
- No data from server: confirm the reference server is running and reachable; check that `upload` requests succeed.
