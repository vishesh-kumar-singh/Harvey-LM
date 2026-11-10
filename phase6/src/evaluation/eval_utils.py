"""
Phase 6: Evaluation & Submission Packaging Utility.

Implements exact metric calculations matching course script evaluate_phase6.py:
- Primary: BERTScore F1
- Secondary: ROUGE-1 F1, ROUGE-L F1
- Submission export and format verification.
"""

import os
import json
import zipfile
import torch
from typing import Dict, List, Tuple, Optional
from rouge_score import tokenizers, rouge_scorer
import bert_score


def lcs(a, b):
    """Longest Common Subsequence matching evaluate_phase6.py."""
    if len(a) > len(b):
        a, b = b, a
    masks = {}
    for i, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << i)
    s = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | s
        y = (s << 1) | 1
        s = x & ~(x - y)
    return s.bit_count() if hasattr(int, 'bit_count') else bin(s).count('1')


def compute_summarization_metrics(
    predictions: Dict[str, str],
    references: Dict[str, str],
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Computes BERTScore F1, ROUGE-1 F1, and ROUGE-L F1.
    """
    common_ids = sorted(list(set(references.keys()) & set(predictions.keys())))
    assert len(common_ids) > 0, "No common IDs between predictions and references!"
    preds = [predictions[k].strip() for k in common_ids]
    refs = [references[k].strip() for k in common_ids]

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    tok = tokenizers.DefaultTokenizer(use_stemmer=True)

    r1_total = rl_total = 0.0
    for pr, rf in zip(preds, refs):
        s = scorer.score(rf, pr)
        r1_total += s['rouge1'].fmeasure

        pt = tok.tokenize(pr)
        rt = tok.tokenize(rf)
        if not pt and not rt:
            x = 1.0
        elif not pt or not rt:
            x = 0.0
        else:
            z = lcs(pt, rt)
            pp = z / len(pt)
            rr = z / len(rt)
            x = 0.0 if (pp + rr == 0) else (2 * pp * rr / (pp + rr))
        rl_total += x

    n = len(common_ids)
    r1 = r1_total / n
    rl = rl_total / n

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # BERTScore
    _, _, f = bert_score.score(preds, refs, lang='en', batch_size=16, verbose=False, device=device)
    bert_f1 = float(f.mean().item())

    return {
        "bertscore_f1": bert_f1,
        "rouge1_f1": r1,
        "rougel_f1": rl,
        "num_docs": n
    }


def export_phase6_submission(predictions: Dict[str, str], output_dir: str) -> str:
    """
    Exports predictions as prediction.json inside submission.zip directly at root.
    Uses official List format recommended by course brief:
      [{"id": "doc_id", "summary": "..."}]
    """
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "prediction.json")
    zip_path = os.path.join(output_dir, "submission.zip")

    # Format as list of objects
    records = [{"id": str(k), "summary": str(v).strip()} for k, v in predictions.items()]

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Zip directly at root
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")

    # Verification
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "prediction.json" in namelist, "prediction.json missing from zip!"
        assert len(namelist) == 1, f"Zip contains unexpected files: {namelist}"

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"Verified submission package: {zip_path} ({len(records)} records, {size_kb:.1f} KB)")
    return zip_path
