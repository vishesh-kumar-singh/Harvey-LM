import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-100, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none', ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        
    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        mask = (labels != self.ignore_index).float()
        return (focal_loss * mask).sum() / mask.sum().clamp(min=1.0)


# Import base model
from phase2.src.model.architecture import HarveyLM, HarveyTransformerConfig

class HarveyForTokenClassification(nn.Module):
    def __init__(self, config: HarveyTransformerConfig, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        
        # Base LM without the LM head
        self.harvey = HarveyLM(config)
        
        # Token classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.dim, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # We need the hidden states from the last layer before the LM head
        kwargs["output_hidden_states"] = True
        
        outputs = self.harvey(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # outputs.hidden_states is a tuple with (h,) containing the normalized hidden state
        hidden_states = outputs.hidden_states[0]
        
        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # We can optionally pass alpha here if we pass it into HarveyForTokenClassification
            # But for simplicity, if self has alpha, we pass it.
            alpha = getattr(self, 'class_weights', None)
            loss_fct = FocalLoss(gamma=2.0, ignore_index=-100, alpha=alpha)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None
        )
