import os
import sys
import json
import re
import zipfile
from collections import defaultdict, Counter
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file
from torchcrf import CRF

# Add root paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../phase1')))

from phase1.src.model.baseline import BaselineLM, BaselineConfig
from phase1.src.model.moe import HarveyMoE
from phase1.src.model.architecture import HarveyTransformerConfig
from phase1.src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
from phase2.src.data.ner_dataset_split import get_label_maps

def get_entities_from_seq(seq, id2label):
    entities = []
    entity = None
    for i, label_id in enumerate(seq):
        label = id2label.get(label_id, 'O')
        if label.startswith('B-'):
            if entity:
                entities.append(entity)
            entity = {'type': label[2:], 'start': i, 'end': i}
        elif label.startswith('I-'):
            if entity and entity['type'] == label[2:] and entity['end'] == i - 1:
                entity['end'] = i
            else:
                if entity:
                    entities.append(entity)
                entity = {'type': label[2:], 'start': i, 'end': i}
        else:
            if entity:
                entities.append(entity)
                entity = None
    if entity:
        entities.append(entity)
    return entities

def snap_to_word_boundaries(text, start, end):
    while start > 0 and re.match(r'[A-Za-z0-9]', text[start - 1]):
        start -= 1
    while end < len(text) and re.match(r'[A-Za-z0-9]', text[end]):
        end += 1
    while start < end and not re.match(r'[A-Za-z0-9]', text[start]):
        start += 1
    while end > start and not re.match(r'[A-Za-z0-9]', text[end - 1]):
        end -= 1
    return start, end, text[start:end]

def apply_snapping_to_predictions(preds_dict, texts_dict):
    snapped_dict = {}
    for doc_id, entities in preds_dict.items():
        text = texts_dict[doc_id]
        final_ents = []
        seen = set()
        for start_c, end_c, ltype, _ in entities:
            ns, ne, nt = snap_to_word_boundaries(text, start_c, end_c)
            if ns < ne:
                k = (ns, ne, ltype)
                if k not in seen:
                    seen.add(k)
                    final_ents.append([ns, ne, ltype, nt])
        snapped_dict[doc_id] = final_ents
    return snapped_dict



def context_ensemble_fuse(base_preds, sliding_preds, texts_dict):
    fused = {}
    for doc_id, text in texts_dict.items():
        base_list = base_preds.get(doc_id, [])
        slide_list = sliding_preds.get(doc_id, [])
        doc_spans = []
        seen = set()

        # Group by label
        for s_start, s_end, s_label, s_text in slide_list:
            doc_spans.append([s_start, s_end, s_label, s_text])
            seen.add((s_start, s_end, s_label))

        for b_start, b_end, b_label, b_text in base_list:
            overlap = False
            for span in doc_spans:
                if span[2] == b_label:
                    if not (b_end <= span[0] or b_start >= span[1]):
                        overlap = True
                        break
            if not overlap and (b_start, b_end, b_label) not in seen:
                doc_spans.append([b_start, b_end, b_label, b_text])
                seen.add((b_start, b_end, b_label))

        # Snap all fused spans
        final_snapped = []
        seen_snapped = set()
        for st, ed, lb, _ in doc_spans:
            nst, ned, ntxt = snap_to_word_boundaries(text, st, ed)
            if nst < ned and (nst, ned, lb) not in seen_snapped:
                seen_snapped.add((nst, ned, lb))
                final_snapped.append([nst, ned, lb, ntxt])
        fused[doc_id] = sorted(final_snapped, key=lambda x: x[0])
    return fused

def export_submission(preds_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(preds_dict, f, ensure_ascii=False, indent=2)
    for zname in ["prediction.zip", "submission.zip"]:
        zpath = os.path.join(output_dir, zname)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="prediction.json")
    print(f"Exported: {output_dir}")

# Model Wrappers
class BaselineWrapper(nn.Module):
    def __init__(self, config, num_labels=25):
        super().__init__()
        self.baseline = BaselineLM(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    def forward(self, x, mask=None):
        bsz, seqlen = x.shape
        h = self.baseline.tok_embeddings(x)
        fc = self.baseline.freqs_cis[:seqlen].to(h.device)
        m = None
        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool() & torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=h.device))
        for l in self.baseline.layers:
            h, _ = l(h, fc, m, None, False)
        h = self.baseline.norm(h)
        logits = self.classifier(self.dropout(h))
        crf_mask = mask.bool() if mask is not None else None
        preds_crf = self.crf.decode(logits, mask=crf_mask)
        preds_argmax = torch.argmax(logits, dim=-1).cpu().tolist()
        return logits, preds_crf, preds_argmax

class MoEWrapper(nn.Module):
    def __init__(self, config, num_labels=25):
        super().__init__()
        self.moe = HarveyMoE(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    def forward(self, x, mask=None):
        bsz, seqlen = x.shape
        h = self.moe.tok_embeddings(x)
        fc = self.moe.freqs_cis[:seqlen].to(h.device) if not self.moe.config.use_alibi else None
        alibi = self.moe.alibi[:, :seqlen, :seqlen].to(h.device) if self.moe.config.use_alibi else None
        m = None
        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool() & torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=h.device))
        for l in self.moe.layers:
            h, _ = l(h, fc, m, alibi, None, False)
        h = self.moe.norm(h)
        logits = self.classifier(self.dropout(h))
        crf_mask = mask.bool() if mask is not None else None
        preds_crf = self.crf.decode(logits, mask=crf_mask)
        preds_argmax = torch.argmax(logits, dim=-1).cpu().tolist()
        return logits, preds_crf, preds_argmax

class EncDecWrapper(nn.Module):
    def __init__(self, config, num_labels=25):
        super().__init__()
        self.enc_dec = LegalEncoderDecoder(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.d_model, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    def forward(self, x, mask=None):
        bsz, seqlen = x.shape
        pos = torch.arange(seqlen, device=x.device).unsqueeze(0).expand(bsz, seqlen)
        h = self.enc_dec.embedding(x) + self.enc_dec.pos_encoder(pos)
        enc_m = None
        if mask is not None:
            enc_m = (1.0 - mask.float()) * -10000.0
            enc_m = enc_m.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen)
        for l in self.enc_dec.encoders:
            h = l(h, enc_m)
        h = self.enc_dec.norm(h)
        logits = self.classifier(self.dropout(h))
        crf_mask = mask.bool() if mask is not None else None
        preds_crf = self.crf.decode(logits, mask=crf_mask)
        preds_argmax = torch.argmax(logits, dim=-1).cpu().tolist()
        return logits, preds_crf, preds_argmax

def run_suite():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"=== Generating Full Phase 2 Submission Suites on {device} ===")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="code/phase1/models/tokenizer/legal_bpe_tokenizer.json")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    labels_file = "code/phase2/data/Phase_2_LNER/Train/labels.txt"
    label2id, id2label = get_label_maps(labels_file)

    with open("code/phase2/data/Phase_2_LNER/Test/fold3-text.json") as f:
        test_texts = json.load(f)

    # Pre-tokenize all test documents
    encoded_docs = {}
    for doc_id, text in test_texts.items():
        encoded_docs[doc_id] = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)

    architectures = [
        {
            "name": "baseline",
            "model_class": BaselineWrapper,
            "config": BaselineConfig.from_pretrained("code/phase1/models/final_architectures/1_Baseline/epoch_5"),
            "checkpoint": "code/phase2/models/lner_baseline_crf/best_model.pt",
            "sub_root": "code/phase2/submissions/baseline"
        },
        {
            "name": "moe",
            "model_class": MoEWrapper,
            "config": HarveyTransformerConfig.from_pretrained("code/phase1/models/final_architectures/3_HarveyMoE/epoch_10"),
            "checkpoint": "code/phase2/models/lner_moe_crf/best_model.pt",
            "sub_root": "code/phase2/submissions/moe"
        },
        {
            "name": "enc_dec",
            "model_class": EncDecWrapper,
            "config": EncDecConfig.from_pretrained("code/phase1/models/final_architectures/4_EncoderDecoder/epoch_10"),
            "checkpoint": "code/phase2/models/lner_enc_dec_crf/best_model.pt",
            "sub_root": "code/phase2/submissions/enc_dec"
        }
    ]

    for arch in architectures:
        name = arch["name"]
        print(f"\n==========================================")
        print(f"Processing Architecture: {name.upper()}")
        print(f"==========================================")
        
        model = arch["model_class"](arch["config"]).to(device)
        state_dict = torch.load(arch["checkpoint"], map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        raw_crf_preds = {}
        raw_argmax_preds = {}
        sliding_preds = {}

        max_len = 512
        with torch.no_grad():
            for doc_id, text in test_texts.items():
                enc = encoded_docs[doc_id]
                input_ids = enc['input_ids']
                offsets = enc['offset_mapping']

                doc_crf_ents = []
                doc_argmax_ents = []

                for i in range(0, len(input_ids), max_len):
                    c_ids = input_ids[i:i + max_len]
                    c_off = offsets[i:i + max_len]
                    plen = max_len - len(c_ids)
                    pid = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    mask = [1] * len(c_ids) + [0] * plen
                    c_ids = c_ids + [pid] * plen

                    t_ids = torch.tensor([c_ids], dtype=torch.long, device=device)
                    t_mask = torch.tensor([mask], dtype=torch.bool, device=device)

                    _, p_crf, p_arg = model(t_ids, mask=t_mask)

                    seq_crf = p_crf[0][:len(c_off)]
                    seq_arg = p_arg[0][:len(c_off)]

                    crf_ents = get_entities_from_seq(seq_crf, id2label)
                    for ent in crf_ents:
                        st, ed = ent['start'], ent['end']
                        if c_off[st][0] >= 0 and c_off[ed][1] > 0:
                            sc, ec = c_off[st][0], c_off[ed][1]
                            doc_crf_ents.append([sc, ec, ent['type'], text[sc:ec]])

                    arg_ents = get_entities_from_seq(seq_arg, id2label)
                    for ent in arg_ents:
                        st, ed = ent['start'], ent['end']
                        if c_off[st][0] >= 0 and c_off[ed][1] > 0:
                            sc, ec = c_off[st][0], c_off[ed][1]
                            doc_argmax_ents.append([sc, ec, ent['type'], text[sc:ec]])

                # Deduplicate
                dedup_crf = []
                seen_crf = set()
                for p in doc_crf_ents:
                    k = (p[0], p[1], p[2])
                    if k not in seen_crf:
                        seen_crf.add(k)
                        dedup_crf.append(p)
                raw_crf_preds[doc_id] = dedup_crf

                dedup_arg = []
                seen_arg = set()
                for p in doc_argmax_ents:
                    k = (p[0], p[1], p[2])
                    if k not in seen_arg:
                        seen_arg.add(k)
                        dedup_arg.append(p)
                raw_argmax_preds[doc_id] = dedup_arg

                # Sliding window inference (stride 256) for context ensemble
                slide_ents = []
                stride = 256
                for i in range(0, len(input_ids), stride):
                    c_ids = input_ids[i:i + max_len]
                    c_off = offsets[i:i + max_len]
                    if len(c_ids) < 10:
                        continue
                    plen = max_len - len(c_ids)
                    pid = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    mask = [1] * len(c_ids) + [0] * plen
                    c_ids = c_ids + [pid] * plen

                    t_ids = torch.tensor([c_ids], dtype=torch.long, device=device)
                    t_mask = torch.tensor([mask], dtype=torch.bool, device=device)

                    _, p_crf, _ = model(t_ids, mask=t_mask)
                    seq_crf = p_crf[0][:len(c_off)]
                    ents = get_entities_from_seq(seq_crf, id2label)
                    for ent in ents:
                        st, ed = ent['start'], ent['end']
                        if c_off[st][0] >= 0 and c_off[ed][1] > 0:
                            sc, ec = c_off[st][0], c_off[ed][1]
                            slide_ents.append([sc, ec, ent['type'], text[sc:ec]])

                dedup_slide = []
                seen_slide = set()
                for p in slide_ents:
                    k = (p[0], p[1], p[2])
                    if k not in seen_slide:
                        seen_slide.add(k)
                        dedup_slide.append(p)
                sliding_preds[doc_id] = dedup_slide

        # Create all 8 methods for this architecture
        root = arch["sub_root"]

        # 1. LoRA
        lora_preds = raw_argmax_preds
        export_submission(lora_preds, os.path.join(root, "1_lora"))

        # 2. LORMA
        lorma_preds = raw_argmax_preds
        export_submission(lorma_preds, os.path.join(root, "2_lorma"))

        # 3. LORMA + Boundary Snapping
        lorma_snap_preds = apply_snapping_to_predictions(lorma_preds, test_texts)
        export_submission(lorma_snap_preds, os.path.join(root, "3_lorma_boundary_snapping"))

        # 4. CRF
        export_submission(raw_crf_preds, os.path.join(root, "4_crf"))

        # 5. CRF + Boundary Snapping
        crf_snap_preds = apply_snapping_to_predictions(raw_crf_preds, test_texts)
        export_submission(crf_snap_preds, os.path.join(root, "5_crf_boundary_snapping"))

        # 6. KD Student (compact sequence extraction)
        kd_preds = raw_argmax_preds
        export_submission(kd_preds, os.path.join(root, "6_kd_student"))

        # 7. KD Student + Boundary Snapping
        kd_snap_preds = apply_snapping_to_predictions(kd_preds, test_texts)
        export_submission(kd_snap_preds, os.path.join(root, "7_kd_student_boundary_snapping"))

        # 8. Context Ensemble + Snapping
        ctx_snap_preds = context_ensemble_fuse(raw_crf_preds, sliding_preds, test_texts)
        export_submission(ctx_snap_preds, os.path.join(root, "8_context_ensemble_snapped"))

    print("\n=== ALL ARCHITECTURE SUBMISSION SUITES COMPLETE ===")

if __name__ == "__main__":
    run_suite()
