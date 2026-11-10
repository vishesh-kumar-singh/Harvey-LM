"""
Phase 3 - Document Sequential Viterbi Decoder for Rhetorical Role Prediction.

In legal court judgments, rhetorical roles exhibit strong sequential continuity
(e.g., Facts cluster in long blocks, Arguments transition into Statute/Precedent,
and Ratios precede Rulings). Empirical self-transition probabilities are 85-91%.

This decoder combines sentence emission log-probabilities from HarveyLM with the
empirical legal transition prior via dynamic programming (Viterbi decoding):

    score(y_{1:N}) = sum_{t=1}^N log P(s_t | y_t) + lambda * sum_{t=2}^N log P(y_t | y_{t-1})
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F


class LegalViterbiDecoder:
    """
    Viterbi decoder incorporating legal rhetorical role transition priors.
    """
    def __init__(self, train_files, label2id, lambda_trans=0.35, min_prob=1e-6):
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.lambda_trans = lambda_trans
        self.min_prob = min_prob
        self.log_trans = self._estimate_transitions(train_files)

    def _estimate_transitions(self, train_files):
        counts = np.ones((self.num_classes, self.num_classes), dtype=np.float64) * 0.1  # Laplace smoothing
        for fpath in train_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                docs = json.load(f)
            for d in docs:
                labels = d.get("labels", [])
                if not labels:
                    continue
                label_ids = [self.label2id.get(l, 0) for l in labels]
                for i in range(1, len(label_ids)):
                    counts[label_ids[i - 1], label_ids[i]] += 1.0

        # Row-normalize to transition probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        prob = counts / row_sums
        prob = np.clip(prob, self.min_prob, 1.0)
        return np.log(prob)

    def decode_document(self, logits) -> list:
        """
        Runs Viterbi decoding over a single document's sentence logits.
        logits: shape (N_sents, num_classes) - np.ndarray or torch.Tensor
        Returns: list of integer class predictions of length N_sents
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().float().numpy()
        n_sents, k = logits.shape
        if n_sents == 0:
            return []
        if n_sents == 1:
            return [int(np.argmax(logits[0]))]

        # Log-softmax emissions
        emissions = logits - np.max(logits, axis=-1, keepdims=True)
        emissions = emissions - np.log(np.sum(np.exp(emissions), axis=-1, keepdims=True) + 1e-12)

        viterbi = np.zeros((n_sents, k), dtype=np.float64)
        backpointer = np.zeros((n_sents, k), dtype=np.int32)

        # Initial step
        viterbi[0] = emissions[0]

        # Forward DP
        trans = self.log_trans * self.lambda_trans
        for t in range(1, n_sents):
            # Broadcast: viterbi[t-1] is (K, 1), trans is (K, K)
            prev_scores = viterbi[t - 1, :, None] + trans  # (K, K): prev -> curr
            best_prev = np.argmax(prev_scores, axis=0)      # (K,)
            best_val  = np.max(prev_scores, axis=0)         # (K,)

            viterbi[t] = best_val + emissions[t]
            backpointer[t] = best_prev

        # Backtrack
        best_path = [int(np.argmax(viterbi[-1]))]
        for t in range(n_sents - 1, 0, -1):
            prev_label = backpointer[t, best_path[-1]]
            best_path.append(int(prev_label))

        best_path.reverse()
        return best_path
