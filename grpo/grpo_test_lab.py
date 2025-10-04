import torch
import re
from math_verify import parse, verify, ExprExtractionConfig

# import lovely_tensors as lt
# lt.monkey_patch()

def get_per_token_logps(logits, input_ids):
    """CPU version of get_per_token_logps for testing."""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def reward_correct(item, answer):
    """Same as in grpo_lab.py."""
    # TODO: Implement reward_correct function
    answer_phrase = re.search(r"<answer>\s*(.*?)\s*</answer>", answer, flags=re.DOTALL)
    answer_phrase = answer_phrase.group(1) if answer_phrase else ""
    number_at_end_re = r'([+-]?(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*$'
    number_at_end = re.search(number_at_end_re, answer_phrase)

    if number_at_end:
        model_answer = parse(number_at_end.group(0))
    else:
        return -1

    gt_answer = parse(item['A'])
    is_correct = verify(gt_answer, model_answer)
    
    if is_correct:
        return 1
    else:
        return -1

def reward_format(item, answer):
    """Same as in grpo_lab.py."""
    think_phrase = re.findall(r"<think>\s*(.*?)\s*</think>", answer, flags=re.DOTALL)
    if len(think_phrase) > 1 or len(think_phrase) == 0:
        return -1

    answer_phrase = re.findall(r"<answer>\s*(.*?)\s*</answer>", answer, flags=re.DOTALL)
    if len(answer_phrase) > 1 or len(answer_phrase) == 0:
        return -1

    return 1.25


def recompute_grpo_loss(vars_dict):
    """Recompute GRPO loss on CPU using saved variables."""
    # Global Variables
    pad_token_id = vars_dict['pad_token_id']
    clip_param = vars_dict['clip_param']
    beta = vars_dict['beta']
    compute_gen_logps = vars_dict['compute_gen_logps']
    
    # Local Variables
    prompt_length = vars_dict['prompt_length']
    inputs = vars_dict['inputs']
    advantages = vars_dict['advantages']
    logits = vars_dict['logits']
    # Align with GRPO_step: slice logits to exclude the last position, and align input_ids to next token
    B, L, V = logits.shape
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    # Ensure time dimensions match (some models emit logits with length inputs-1 already)
    T = min(logits.shape[1], input_ids.shape[1])
    if logits.shape[1] != input_ids.shape[1]:
        print(f"[GRPO] Aligning time dims: logits.T={logits.shape[1]} input_ids.T={input_ids.shape[1]} -> T={T}")
    logits = logits[:, :T, :]
    input_ids = input_ids[:, :T]

    # Optional: decode a small window of aligned tokens
    try:
        from transformers import AutoTokenizer
        import os
        tok_name = os.environ.get("GRPO_TOKENIZER", "Qwen/Qwen2.5-3B")
        tokenizer = AutoTokenizer.from_pretrained(tok_name)

        p = int(prompt_length) if not isinstance(prompt_length, (list, tuple)) else int(prompt_length[0])
        t0 = max(0, p - 3)
        t1 = min(T, p + 5)

        print(f"[GRPO][decode] window t in [{t0}, {t1}) (T={T}, prompt_length={p})")
        ids_in  = inputs[0, t0:t1].tolist()         # tokens observed at time t
        ids_tgt = input_ids[0, t0:t1].tolist()      # target tokens at time t (i.e., inputs[t+1])
        preds   = logits[0, t0:t1, :].argmax(-1).tolist()  # model's top-1 prediction

        toks_in  = tokenizer.convert_ids_to_tokens(ids_in)
        toks_tgt = tokenizer.convert_ids_to_tokens(ids_tgt)
        toks_pred= tokenizer.convert_ids_to_tokens(preds)

        # Optional: show context text up to t1
        print("[GRPO][decode] context: \n", tokenizer.decode(inputs[0, :t1], skip_special_tokens=False))
        for i, t in enumerate(range(t0, t1)):
            print(f" t={t:>3}: inputs={toks_in[i]!r} -> input_ids={toks_tgt[i]!r} logits={toks_pred[i]!r}")

    except Exception as e:
        print(f"[GRPO][decode] skipped: {e}")

    # Accept either key name from saved vars
    refs_per_token_logps = vars_dict.get('refs', None) # pi_ref
    if refs_per_token_logps is None:
        refs_per_token_logps = vars_dict['ref_per_token_logps']
    gen_logps = vars_dict.get('gen_logps', None) if compute_gen_logps else None # pi_old
    assert gen_logps is not None
    
    # Debug: print shapes and small content slices
    try:
        print(f"[GRPO] prompt_length: {int(prompt_length) if not isinstance(prompt_length, (list, tuple)) else prompt_length}")
    except Exception:
        print(f"[GRPO] prompt_length (raw): {prompt_length}")
    print(f"[GRPO] inputs.shape={inputs.shape}, device={inputs.device}, dtype={inputs.dtype}")

    try:
        print(f"[GRPO] inputs[0,:8]={inputs[0, :min(8, inputs.shape[1])].tolist()}")
    except Exception as e:
        print(f"[GRPO] inputs preview error: {e}")
    print(f"[GRPO] logits.shape(before shift)={(B, L, V)}")
    print(f"[GRPO] logits.shape(after shift)={logits.shape}; input_ids.shape={input_ids.shape}")
    try:
        print(f"[GRPO] input_ids[0,:8]={input_ids[0, :min(8, input_ids.shape[1])].tolist()}")
    except Exception as e:
        print(f"[GRPO] input_ids preview error: {e}")
    try:
        print(f"[GRPO] refs_per_token_logps.shape={getattr(refs_per_token_logps, 'shape', None)}")
        if hasattr(refs_per_token_logps, 'shape') and refs_per_token_logps.shape[0] > 0:
            print(f"[GRPO] refs_per_token_logps[0,:8]={refs_per_token_logps[0, :min(8, refs_per_token_logps.shape[1])].tolist()}")
    except Exception as e:
        print(f"[GRPO] refs_per_token_logps preview error: {e}")
    try:
        print(f"[GRPO] gen_logps.shape={getattr(gen_logps, 'shape', None)}")
        if hasattr(gen_logps, 'shape') and gen_logps.shape[0] > 0:
            print(f"[GRPO] gen_logps[0,:8]={gen_logps[0, :min(8, gen_logps.shape[1])].tolist()}")
    except Exception as e:
        print(f"[GRPO] gen_logps preview error: {e}")
    
    # 1. Compute per-token log probabilities of the model
    # 2. Slice to keep only completion tokens (after prompt)
    # 3. Move reference log probabilities to the same device as per_token_logps
    # 4. Compute per-token KL divergence approximation for regularization
    # 5. Create mask for completion tokens (not padding)
    # 6. Compute importance sampling ratio
    # 7. Clip ratio for PPO-style loss
    # 8. Compute per-token GRPO loss
    # 9. Average loss over completion tokens and batch
    # 10. Return final loss 

    # 1. Compute per-token log probabilities of the model
    per_token_logps = get_per_token_logps(logits, input_ids)

    # 2. Slice to keep only completion tokens (after prompt)
    s = int(prompt_length) - 1
    R = refs_per_token_logps.shape[1]
    avail = per_token_logps.shape[1] - s
    Lc = max(0, min(R, avail))                       # [B, Lc]
    
    per_token_answer_logps = per_token_logps[:, s:s+Lc] # pi_theta
    targets = input_ids[:, s:s+Lc] # ground truth tokens

    # 3. Move reference log probabilities to the same device as per_token_logps
    device = per_token_answer_logps.device
    refs_per_token_logps = refs_per_token_logps.to(device)[:, :Lc] # to(engine.device) 
    gen_logps = gen_logps.to(device)[:, :Lc]
    advantages = advantages.to(device)
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)                        # [B, 1]

    # 4. Compute per-token KL divergence approximation for regularization
    delta = refs_per_token_logps - per_token_answer_logps
    kl_div = torch.exp(delta) - delta - 1
    
    # 5. Create mask for completion tokens (not padding)
    completion_mask = (targets != pad_token_id).float()

    # 6. Compute importance sampling ratio
    ratio = torch.exp(per_token_answer_logps - gen_logps) # because they are log probabilities, subtracting log(pi_theta) - log(pi_old) is the same as log(pi_theta / pi_old)

    # 7. Clip ratio for PPO-style loss
    clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)

    # 8. Compute per-token GRPO loss
    per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # 9. Average loss over completion tokens and batch
    sequence_denom = completion_mask.sum(dim=1).clamp_min(1.0)
    sequence_avg_per_token_loss = (per_token_loss * completion_mask).sum(dim=1) / sequence_denom

    # 10. Return final loss 
    return -sequence_avg_per_token_loss.mean()

if __name__ == '__main__':
    print("Running GRPO lab tests...")
    print("=" * 40)
    print("Recomputing GRPO loss from saved variables...")
    # Load saved variables
    vars_dict = torch.load('grpo_step_vars.pth', weights_only=True)
    expected_loss = vars_dict['loss']
    
    # Recompute loss on CPU
    computed_loss = recompute_grpo_loss(vars_dict)
    
    # Harmonize dtypes/devices before comparison
    exp = expected_loss.detach().cpu().float()
    comp = computed_loss.detach().cpu().float()
    print(f"Expected Loss: {exp.item():.6f}")
    print(f"Computed Loss: {comp.item():.6f}")
    print(f"Match: {torch.allclose(exp, comp, atol=1e-3)}")
    print("=" * 40)
    
    # Tests for reward functions
    test_item = {'Q': 'What is 2+2?', 'A': '4'}
    
    # Test reward_correct
    print(f"Testing reward_correct...")
    assert reward_correct(test_item, '<think>2+2=4</think><answer>4</answer>') == 1
    assert reward_correct(test_item, '<think>2+2=5</think><answer>5</answer>') == -1
    assert reward_correct(test_item, 'No number') == -1
    print("reward_correct tests passed")
    print("=" * 40)
    
    # Test reward_format
    print(f"Testing reward_format...")
    assert reward_format(test_item, '<think>reasoning</think><answer>answer</answer>') == 1.25
    assert reward_format(test_item, '<think>reasoning</think> <answer>answer</answer>') == 1.25  # with space
    assert reward_format(test_item, '<think>reasoning<answer>answer</answer>') == -1  # missing </think>
    assert reward_format(test_item, '<think>reasoning</think><answer>answer</answer><think>extra</think>') == -1  # extra tags
    print("reward_format tests passed")
    print("=" * 40)
