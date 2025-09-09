"""
GRPO Lab

Goal: Build the core GRPO-style policy update for LLMs with a reference KL.

In this lab, you will implement the essential components while we provide
the boilerplate around them. 

What you will implement (TODOs in this file):
1) reward_format(answer: str) -> float
   - Return +1.25 if output matches the exact format
     <think>...</think><answer>...</answer> (one pair each), else -1.0.

2) reward_correct(item: dict, answer: str) -> float
   - Extract the final numeric answer (int/float/fraction) from the model output
     and compare against item["A"]. Return +1.0 if correct, else -1.0.

3) gen_worker(...) #Core PPO inference
   - Use vLLM to generate a single rollout per prompt (PPO uses one rollout).
   - Compute rewards (format + correctness) per rollout.
   - Compute on-policy gen_logps for the generated tokens via vLLM by passing
     the concatenated [prompt || completion] token ids back as prompt_token_ids
     and requesting prompt_logprobs.
   - Package tensors and POST to the reference server /upload.
   - Later (Task 5), you will add GRPO group-normalization to rewards and enable GROUP_SIZE > 1.

4) PPO_step(batch, ...) #Loss computation which we backprop through
   - Implement the PPO ratio path using gen_logps.
   - Compute completion-only per-token logps, mask padding, add reference KL
     penalty, and return the scalar mean loss.

5) normalize_group_rewards(rewards: torch.Tensor) -> torch.Tensor (GRPO)
   - Turn PPO into GRPO:
     - Add a GROUP_SIZE variable in the Config section (start with GROUP_SIZE = 2 or 3).
     - Update gen_worker to generate multiple rollouts per prompt (SamplingParams n=GROUP_SIZE),
       adjust per-prompt answer collection, batching, and JSON headers.
     - Normalize rewards within each group: (rewards - mean) / (std + 1e-4).

Run modes:
- Reference server (provides reference per-token log-probs over completions):
    python grpo_lab_ppo.py --ref-server
- Training (spawns generation worker on rank 0):
    deepspeed --num_gpus <N> grpo_lab_ppo.py --train
"""

import os, json, re, random, time, io
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# =========================
# Config
# =========================

MODEL_PATH = "Qwen/Qwen2.5-3B"
REF_SERVER_URL = "http://localhost:59875"

GEN_DEVICE = 0
# NOTE: Do not add GROUP_SIZE in PPO steps. You will introduce GROUP_SIZE in Step 5.
TRAIN_BATCH_SIZE = 1

BETA = 0.04
CLIP_PARAM = 0.2
ALL_STEPS = 1000
SAVE_STEPS = 50
GEN_UPDATE_STEPS = 3

DS_CONFIG = {
    "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}


# =========================
# Serialization helpers
# =========================

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()


def bytes_to_tensor(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)


def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()


def bytes_list_to_list(b: bytes):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist


# =========================
# Reference server (implemented)
# =========================

def run_reference_server(model_path: str = "Qwen/Qwen2.5-1.5B-Instruct", port: int = 59875):
    import bottle, threading, queue
    from bottle import request

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, _attn_implementation="sdpa"
    ).to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids: torch.Tensor) -> torch.Tensor:
        logits = ref_model(input_ids).logits
        logits = logits[:, :-1, :]
        input_ids_shift = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, ids_row in zip(logits, input_ids_shift):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, 1, ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        if len(dd) not in (3, 4):
            return b'tensor'
        data = {'base': json.loads(dd[0])}
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        if len(dd) == 4:
            data['gen_logps'] = bytes_to_tensor(dd[3])
        raw_queue.put(data)
        return b'tensor'

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty():
            return b'empty'
        return result_queue.get()

    def run_server():
        bottle.run(app, host='0.0.0.0', port=port, server='tornado')

    threading.Thread(target=run_server, daemon=False).start()

    while True:
        d = raw_queue.get()
        prompt_length = d['base']['plen']
        with torch.inference_mode():
            per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        per_token_logps = per_token_logps.to(torch.float16)
        data = [
            json.dumps(d['base']).encode(),
            tensor_to_bytes(d['inputs']),
            tensor_to_bytes(d['rewards']),
            tensor_to_bytes(per_token_logps),
        ]
        if 'gen_logps' in d:
            data.append(tensor_to_bytes(d['gen_logps']))
        xdata = make_bytes_list(data)
        result_queue.put(xdata)


# =========================
# Rewards (TODOs)
# =========================

def reward_format(answer: str) -> float:
    """
    TODO (Task 1a): Implement the format reward.

    Requirements:
    - Return +1.25 if the answer matches exactly one pair of <think>...</think>
      and one pair of <answer>...</answer>, allowing whitespace/newlines between
      the two blocks; otherwise return -1.0.
    """
    raise NotImplementedError("Implement reward_format(answer) per the spec.")


def reward_correct(item: dict, answer: str) -> float:
    """
    TODO (Task 1b): Implement the correctness reward.

    Requirements:
    - Extract the final numeric answer from the model's output (last occurrence of
      integer, float, or fraction), compare to item["A"]. Return +1.0 if equal,
      else -1.0. You may parse fractions by converting to float for a baseline.
    """
    raise NotImplementedError("Implement reward_correct(item, answer) per the spec.")


def normalize_group_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """
    TODO (Task 5): Normalize rewards within a group: (rewards - mean) / (std + 1e-4).
    Input shape: [G]. Return tensor of same shape.
    """
    raise NotImplementedError("Implement normalize_group_rewards(rewards).")


# =========================
# Generation worker (TODOs)
# =========================

def gen_worker(process_queue: mp.Queue, physics_device: int, tokenizer: AutoTokenizer):
    """
    TODO (Task 3): Implement the core PPO data path in the generation worker.

    Objective:
    - Use vLLM to generate a single rollout per prompt, compute rewards, and
      compute on-policy gen_logps for PPO. Send data batches to the reference
      server. This lab version removes the non-gen_logps path. You will add
      GRPO group-normalization later in Task 5.

    Steps:
    1) Instantiate vLLM objects:
       - Create LLM(model=MODEL_PATH, gpu_memory_utilization≈0.3, max_model_len≈4096, dtype="float16").
       - Create SamplingParams for generation with n=1, temperature (e.g., 0.9), and max_tokens (e.g., 700).
       - Create SamplingParams for gen_logps with temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1.

    2) Build prompts:
       - Implement build_chat_prompt(question) using tokenizer.apply_chat_template with the provided system prompt.

    3) Generation loop:
       - Sample one item from GSM8K each iteration.
       - Build prompts and call vLLM.generate(prompts_text, sampling_params, use_tqdm=False).
       - Collect a single answer per prompt (one rollout) and its token_ids.

    4) Rewards:
       - For each generated answer, compute reward = reward_correct(item, answer) + reward_format(answer).
       - Keep a copy of the unnormalized rewards for logging in the JSON header.

    5) Merged ids:
       - For each prompt i, tokenize the prompt to get prompt_ids and plen.
       - Concatenate prompt_ids with the completion token ids to form merged_ids of shape [1, plen + Lc].

    6) Compute on-policy gen_logps:
       - Call vLLM.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False).
       - Extract prompt_logprobs from vLLM outputs; slice to completion region by skipping the first plen tokens
         (align with completion tokens only). Convert to a tensor of shape [1, Lc].

    7) Package and upload to reference server:
       - Create header JSON with keys: {"plen": plen, "unnormalized_rewards": list, "format_accuracy": float or None}.
       - Build payload list: [json_header_bytes, tensor_to_bytes(merged_ids), tensor_to_bytes(rewards), tensor_to_bytes(gen_logps)].
       - Wrap with make_bytes_list and POST to f"{REF_SERVER_URL}/upload".

    8) Repeat continuously.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)

    # Helpers provided for you. Implement only the for-loop per instructions below.
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=MODEL_PATH, gpu_memory_utilization=0.3, max_model_len=4096, dtype="float16")
    sampling_params = SamplingParams(n=1, temperature=0.9, max_tokens=700)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    # Dataset: GSM8K train
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]

    system_prompt = (
        "You are a helpful assistant. A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
    )

    def build_chat_prompt(question: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def gen_answers(prompts_text):
        """Return lists: answers (str) and ans_token_ids (list[list[int]])."""
        voutputs = vllm_gen.generate(prompts_text, sampling_params, use_tqdm=False)
        answers, ans_token_ids = [], []
        for v in voutputs:
            for z in v.outputs:
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        return answers, ans_token_ids

    def encode_prompt(prompt_text: str):
        ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids, ids.shape[1]

    def build_merged_ids(prompt_ids: torch.Tensor, completion_token_ids: list[int]) -> torch.Tensor:
        completion = torch.tensor(completion_token_ids, dtype=torch.long).unsqueeze(0)
        return torch.cat([prompt_ids, completion], dim=1)

    def compute_gen_logps_for_merged(merged_ids: torch.Tensor, plen: int) -> torch.Tensor:
        z = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
        # z is a list with one item; slice prompt_logprobs to completion region
        comp = z[0].prompt_logprobs[plen:]
        return torch.tensor([[list(x.values())[0].logprob for x in comp]])

    def upload_payload(plen: int, merged_ids: torch.Tensor, reward_value: float, fmt_ok: float, gen_logps: torch.Tensor):
        header = {
            "plen": plen,
            "unnormalized_rewards": [float(reward_value)],
            "format_accuracy": float(fmt_ok)
        }
        data = [
            json.dumps(header).encode(),
            tensor_to_bytes(merged_ids),
            tensor_to_bytes(torch.tensor([reward_value], dtype=torch.float32)),
            tensor_to_bytes(gen_logps),
        ]
        xdata = make_bytes_list(data)
        requests.post(f"{REF_SERVER_URL}/upload", data=xdata)

    # TODO (Task 3): Implement the generation loop, rewards, gen_logps, and upload using the helpers above.
    while True:
        # IMPLEMENTATION REQUIRED BY STUDENT
        raise NotImplementedError("Implement Task 3 for gen_worker: loop, rewards, gen_logps, upload.")


# =========================
# PPO step (TODO)
# =========================

def per_token_logps_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def PPO_step(batch: dict, engine, tokenizer, beta: float, clip_param: float) -> torch.Tensor:
    """
    TODO (Task 4): Implement PPO loss with reference KL.

    Inputs in batch:
    - 'plen': int, prompt token length
    - 'inputs': LongTensor [B, L] (merged prompt + completion)
    - 'rewards': FloatTensor [B] (group-normalized advantages)
    - 'refs': FloatTensor [B, Lc] (reference per-token log-probs over completion)
    - 'gen_logps': FloatTensor [B, Lc] (on-policy per-token log-probs over completion)

    Steps:
    1) Forward current model; get logits.
    2) Compute per-token logps aligned to input_ids[:, 1:], then slice to completion: [plen-1:].
    3) ratio = exp(curr - gen_logps); clipped_ratio = clamp(ratio, 1 - clip_param, 1 + clip_param)
       per_token_obj = min(ratio * adv, clipped_ratio * adv)
    4) per-token forward-KL vs reference: kl = exp(ref - curr) - (ref - curr) - 1
    5) Mask out padding in completion region (tokenizer.pad_token_id), mean over valid tokens, then mean over batch.
    6) loss = - (per_token_obj - beta * kl).mean_over_tokens_and_batch
    """
    raise NotImplementedError("Implement PPO_step with gen_logps, masking, and ref KL.")


# =========================
# Trainer
# =========================

def trainer_main():
    import deepspeed
    

    deepspeed.init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if dist.get_rank() == 0:
        mp.set_start_method('spawn', force=True)
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, GEN_DEVICE, tokenizer))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
    engine, optimizer, _, _ = deepspeed.initialize(config=DS_CONFIG, model=model, model_parameters=model.parameters())

    

    def get_batch():
        try:
            r = requests.get(f"{REF_SERVER_URL}/get").content
            if r == b'empty':
                return None
        except Exception:
            return None
        dd = bytes_list_to_list(r)
        data = json.loads(dd[0])
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        data['refs'] = bytes_to_tensor(dd[3])
        if len(dd) == 5:
            data['gen_logps'] = bytes_to_tensor(dd[4])
        return data

    progress = range(1, ALL_STEPS + 1)
    if dist.get_rank() == 0:
        progress = tqdm(progress)

    for step in progress:
        batch = get_batch()
        while batch is None:
            if dist.get_rank() == 0:
                time.sleep(1)
            batch = get_batch()

        loss = PPO_step(batch, engine, tokenizer, beta=BETA, clip_param=CLIP_PARAM)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            current_loss = loss.item()
            current_reward = np.mean(batch.get('unnormalized_rewards', [0.0]))
            progress.set_description(f"Loss: {current_loss:.6f}, Avg Reward: {current_reward:.3f}")

        if step % GEN_UPDATE_STEPS == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                _ = engine.module.state_dict()
            dist.barrier()

        if step % SAVE_STEPS == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                save_name = f"./step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()

    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-server", action="store_true", help="Run the reference server only.")
    parser.add_argument("--train", action="store_true", help="Run the trainer.")
    parser.add_argument("--ref-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Reference model path.")
    parser.add_argument("--ref-port", type=int, default=59875, help="Reference server port.")
    args = parser.parse_args()

    if args.ref_server:
        run_reference_server(model_path=args.ref_model, port=args.ref_port)
    elif args.train:
        trainer_main()
    else:
        print("Specify one of: --ref-server or --train")

