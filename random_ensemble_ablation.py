def random_ensemble_mean_embedding(self, gpt2, metaicl_model, test_data, dev_data,
                    num_combinations=100, k=8, seed=42, num_anchors=None):
    import numpy as np
    import torch
    from collections import defaultdict
    from tqdm import tqdm
    import random

    random.seed(seed)
    device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

    all_indices = list(range(len(test_data)))

    # -------- util: sampler --------
    def sample_unique_combinations(all_indices, k, num_combinations, seed=42):
        random.seed(seed)
        seen = set()
        combinations = []
        max_trials = num_combinations * 10
        trials = 0
        while len(combinations) < num_combinations and trials < max_trials:
            comb = tuple(sorted(random.sample(all_indices, k)))
            if comb not in seen:
                seen.add(comb)
                combinations.append(comb)
            trials += 1
        return combinations

    sampled_combinations = sample_unique_combinations(all_indices, k, num_combinations, seed=seed)

    # -------- util: get model input embeddings (works for GPT-2/LLaMA-style HF models) --------
    def _get_input_embeddings(module):
        # Prefer standard HF accessor
        if hasattr(module, "get_input_embeddings"):
            return module.get_input_embeddings()
        # Fallbacks for common structures
        if hasattr(module, "model"):
            mdl = module.model
            if hasattr(mdl, "decoder") and hasattr(mdl.decoder, "embed_tokens"):
                return mdl.decoder.embed_tokens
            if hasattr(mdl, "embed_tokens"):
                return mdl.embed_tokens
        if hasattr(module, "embed_tokens"):
            return module.embed_tokens
        raise RuntimeError("Cannot find embedding layer on the provided model.")

    embed_layer = _get_input_embeddings(metaicl_model.model)

    # -------- NEW: select anchors by closeness to global centroid of combination embeddings --------
    # Each combination -> build its anchor_prompt, get embedding by mean-pooling token embeddings.
    with torch.no_grad():
        comb_prompts = []
        for comb in sampled_combinations:
            prompt = "".join([
                f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in comb
            ])
            comb_prompts.append(prompt)

        # Tokenize all prompts (batched for speed & to keep identical padding behavior)
        tok = self.tokenizer(comb_prompts, return_tensors="pt",
                            padding="max_length", truncation=True, max_length=self.max_length)
        input_ids = tok.input_ids.to(device)  # [C, T]
        # [C, T, D] -> mean over T -> [C, D]
        comb_emb = embed_layer(input_ids).mean(dim=1)

        # Global centroid
        centroid = comb_emb.mean(dim=0, keepdim=True)  # [1, D]
        # L2 distance to centroid
        dists = torch.norm(comb_emb - centroid, p=2, dim=1)  # [C]

        # Decide how many anchors: if None or >= total, use all; else pick the closest 'num_anchors'
        if num_anchors is None or num_anchors >= len(sampled_combinations):
            anchor_indices = torch.arange(len(sampled_combinations), device=device)
        else:
            anchor_indices = torch.topk(-dists, k=num_anchors).indices  # negative for smallest distance

        anchors = [sampled_combinations[int(i)] for i in anchor_indices.tolist()]
        anchor_prompts_cached = {sampled_combinations[i]: comb_prompts[i] for i in range(len(sampled_combinations))}

    self.logger.info(f"number of anchors (embedding-nearest): {len(anchors)}")

    # -------- Step 1: precompute anchor info for EVERY selected anchor --------
    anchor_info = {}
    total_flops = 0

    for anchor in tqdm(anchors, desc="Computing anchor info"):
        anchor_prompt = anchor_prompts_cached[anchor]
        base_losses, base_gradients, _, flops = zip(*[
            self.forward_estim(gpt2, metaicl_model, anchor_prompt, dp, dp["task"], return_loss=True)
            for dp in dev_data
        ])
        total_flops += sum(flops)

        loss_tensor = torch.tensor(base_losses, device=device)  # [len(dev), num_labels]
        # base_gradients: list length=len(dev), each is list length=num_labels of gradient vectors [D]
        grad_tensor = torch.stack([torch.stack(g, dim=0) for g in base_gradients], dim=0)  # [len(dev), num_labels, D]
        anchor_info[anchor] = (anchor_prompt, loss_tensor, grad_tensor)

    # -------- Step 2: for EACH anchor, evaluate ALL combinations (including itself) --------
    point_scores = defaultdict(list)
    all_accs = []

    for anchor in tqdm(anchors, desc="Evaluating all combos per anchor"):
        anchor_prompt, base_loss_tensor, grad_tensor = anchor_info[anchor]

        for comb in sampled_combinations:
            target_prompt = anchor_prompts_cached[comb]

            correct = 0
            for dp_idx, dp in enumerate(dev_data):
                dev_str = f"Input: {dp['input']} Label:"
                delta_P = self.compute_embedding_difference_(
                    gpt2, metaicl_model, anchor_prompt + dev_str, target_prompt + dev_str
                )  # [D]

                # taylor correction per label
                taylor_losses = []
                for j in range(len(base_loss_tensor[dp_idx])):
                    correction = torch.sum(grad_tensor[dp_idx][j] * delta_P).item()
                    approx_loss = base_loss_tensor[dp_idx][j].item() + correction
                    taylor_losses.append(approx_loss)

                pred_id = int(np.argmin(taylor_losses))
                pred = dp["options"][pred_id]
                if pred == dp["output"]:
                    correct += 1

            acc = correct / len(dev_data)
            all_accs.append(acc)

            # assign the combo's acc to each member datapoint
            for idx in comb:
                point_scores[idx].append(acc)

    # -------- Step 3: average per-point score across ALL anchors/combos it appears in --------
    avg_scores = []
    for idx, scores in point_scores.items():
        avg_scores.append((idx, float(sum(scores) / len(scores))))

    avg_scores.sort(key=lambda x: -x[1])
    final_indices = [idx for idx, _ in avg_scores[:k]]
    selected_data = [test_data[i] for i in final_indices]

    mean_acc_overall = float(np.mean(all_accs)) if all_accs else 0.0

    return selected_data, mean_acc_overall, total_flops


def random_ensemble_mean_anchor_result(self, gpt2, metaicl_model, test_data, dev_data,
                    num_combinations=100, k=8, seed=42, num_anchors=None):
    import numpy as np
    import torch
    from collections import defaultdict
    from tqdm import tqdm
    import random

    random.seed(seed)
    device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

    all_indices = list(range(len(test_data)))

    # Efficient sampling
    def sample_unique_combinations(all_indices, k, num_combinations, seed=42):
        random.seed(seed)
        seen = set()
        combinations = []
        max_trials = num_combinations * 10
        trials = 0
        while len(combinations) < num_combinations and trials < max_trials:
            comb = tuple(sorted(random.sample(all_indices, k)))
            if comb not in seen:
                seen.add(comb)
                combinations.append(comb)
            trials += 1
        return combinations

    sampled_combinations = sample_unique_combinations(all_indices, k, num_combinations, seed=seed)

    # Use ALL sampled combinations as anchors (as requested)
    anchors = sampled_combinations if (num_anchors is None or num_anchors >= len(sampled_combinations)) \
        else sampled_combinations[:num_anchors]
    self.logger.info(f"number of anchors (using all sampled when None/large): {len(anchors)}")

    anchor_info = {}
    total_flops = 0

    # ---- Step 1: precompute anchor info for EVERY anchor ----
    for anchor in tqdm(anchors, desc="Computing anchor info"):
        anchor_prompt = "".join([
            f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in anchor
        ])
        base_losses, base_gradients, _, flops = zip(*[
            self.forward_estim(gpt2, metaicl_model, anchor_prompt, dp, dp["task"], return_loss=True)
            for dp in dev_data
        ])
        total_flops += sum(flops)

        loss_tensor = torch.tensor(base_losses, device=device)  # [len(dev), num_labels]
        # base_gradients: list length=len(dev), each is list length=num_labels of gradient vectors [D]
        grad_tensor = torch.stack([torch.stack(g, dim=0) for g in base_gradients], dim=0)  # [len(dev), num_labels, D]
        anchor_info[anchor] = (anchor_prompt, loss_tensor, grad_tensor)

    # ---- Step 2: for EACH anchor, evaluate ALL combinations (including itself) ----
    # accumulate scores per datapoint across ALL (anchor, comb) pairs
    point_scores = defaultdict(list)
    all_accs = []

    for anchor in tqdm(anchors, desc="Evaluating all combos per anchor"):
        anchor_prompt, base_loss_tensor, grad_tensor = anchor_info[anchor]

        for comb in sampled_combinations:
            target_prompt = "".join([
                f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in comb
            ])

            correct = 0
            for dp_idx, dp in enumerate(dev_data):
                dev_str = f"Input: {dp['input']} Label:"
                delta_P = self.compute_embedding_difference_(
                    gpt2, metaicl_model, anchor_prompt + dev_str, target_prompt + dev_str
                )  # [D]

                # taylor correction per label
                taylor_losses = []
                for j in range(len(base_loss_tensor[dp_idx])):
                    correction = torch.sum(grad_tensor[dp_idx][j] * delta_P).item()
                    approx_loss = base_loss_tensor[dp_idx][j].item() + correction
                    taylor_losses.append(approx_loss)

                pred_id = int(np.argmin(taylor_losses))
                pred = dp["options"][pred_id]
                if pred == dp["output"]:
                    correct += 1

            acc = correct / len(dev_data)
            all_accs.append(acc)

            # assign the combo's acc to each member datapoint
            for idx in comb:
                point_scores[idx].append(acc)

    # ---- Step 3: average score for each datapoint across ALL anchors/combos it appears in ----
    avg_scores = []
    for idx, scores in point_scores.items():
        avg_scores.append((idx, float(sum(scores) / len(scores))))

    # sort high to low, pick top-k
    avg_scores.sort(key=lambda x: -x[1])
    final_indices = [idx for idx, _ in avg_scores[:k]]
    selected_data = [test_data[i] for i in final_indices]

    # global mean accuracy across all (anchor, comb)
    mean_acc_overall = float(np.mean(all_accs)) if all_accs else 0.0

    return selected_data, mean_acc_overall, total_flops
