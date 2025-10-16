import random


class Score:
    def __init__(self, logprob: float):
        self.logprob = logprob


def build_mock_data(seed: int = 0):
    random.seed(seed)
    # Two generated answers (samples), each with 5 tokens
    output_ids = [
        [42, 7, 9, 9, 11],
        [3, 3, 5, 7, 42],
    ]

    # zz structure after slicing off prompt: zz[j][i] is a dict of candidate token_id -> Score
    # Ensure the actual token (from output_ids[j][i]) is present among candidates
    zz = []
    vocab_pool = list(set(sum(output_ids, []))) + [1, 2, 4, 6, 8, 10]
    for j, seq in enumerate(output_ids):
        per_pos = []
        for i, tok in enumerate(seq):
            candidates = {}
            # Insert the true token with a recognizable logprob pattern
            candidates[tok] = Score(logprob=-0.1 * (i + 1) - 0.01 * (j + 1))
            # Add a few distractor tokens
            distractors = random.sample(vocab_pool, k=3)
            for dt in distractors:
                if dt == tok:
                    continue
                candidates[dt] = Score(logprob=-2.0 - random.random())
            per_pos.append(candidates)
        zz.append(per_pos)

    return output_ids, zz


def extract_actual_logprobs(output_ids, zz):
    actual_logprobs = []
    for j, seq in enumerate(output_ids):
        row = []
        for i, tok_id in enumerate(seq):
            pos_dict = zz[j][i]
            lp = pos_dict.get(tok_id).logprob if tok_id in pos_dict else None
            row.append(lp)
        actual_logprobs.append(row)
    return actual_logprobs


def main():
    output_ids, zz = build_mock_data()
    actual_logprobs = extract_actual_logprobs(output_ids, zz)

    print("Output token IDs per sample:")
    for j, seq in enumerate(output_ids):
        print(f"  sample {j}: {seq}")

    print("\nAligned per-token logprobs of the actual generated tokens:")
    for j, row in enumerate(actual_logprobs):
        print(f"  sample {j}: {row}")

    # Quick sanity: lengths should match
    assert all(len(a) == len(b) for a, b in zip(output_ids, actual_logprobs)), "Length mismatch"

    # Demonstrate indexing for an arbitrary i
    j, i = 1, 3  # 4th token of sample 2
    tok_id = output_ids[j][i]
    lp = zz[j][i][tok_id].logprob
    print(f"\nExample: sample {j}, token index {i}, token {tok_id}, logprob {lp}")


if __name__ == "__main__":
    main()


