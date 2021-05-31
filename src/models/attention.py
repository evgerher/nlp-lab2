from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from src.models.experiment_setup import AttentionSetup, MODEL_ARCHITECTURE, EmbeddingSetup


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class Attention(nn.Module, ABC):
  def __init__(self, architecture: AttentionSetup,
              enc_architecture: MODEL_ARCHITECTURE,
              dec_architecture: MODEL_ARCHITECTURE):
    super().__init__()
    self.architecture = architecture
    self.enc_architecture = enc_architecture
    self.dec_architecture = dec_architecture

  @abstractmethod
  def forward(self, decoder_embeds, decoder_past_hidden, encoder_states):
    raise NotImplementedError()

  @classmethod
  def resolve(cls, architecture: AttentionSetup,
              enc_architecture: MODEL_ARCHITECTURE,
              dec_architecture: MODEL_ARCHITECTURE,
              enc_emb_setup: EmbeddingSetup,
              dec_emb_setup: EmbeddingSetup) -> 'Attention':
    if architecture.name == 'none':
      return None
    elif architecture.name == 'tutorial':
      return TutorialAttention(architecture, enc_architecture, dec_architecture, enc_emb_setup, dec_emb_setup)


class TutorialAttention(Attention):
  def __init__(self, architecture: AttentionSetup,
              enc_architecture: MODEL_ARCHITECTURE,
              dec_architecture: MODEL_ARCHITECTURE,
              enc_emb_setup: EmbeddingSetup,
              dec_emb_setup: EmbeddingSetup):
    super().__init__(architecture, enc_architecture, dec_architecture)
    self.max_length = enc_emb_setup.max_length

    # todo: make separate argument for output shape
    self.attn = nn.Linear(dec_emb_setup.embedding_size + dec_architecture.hidden_size, self.max_length)
    self.attn_combine = nn.Linear(dec_architecture.hidden_size * 2 + dec_emb_setup.embedding_size, dec_emb_setup.embedding_size)

  def forward(self, decoder_embeds, decoder_past_hidden, encoder_states):
    seq_len = encoder_states.shape[0]
    embed_hidden = torch.cat((decoder_embeds[0], decoder_past_hidden[0]), 1)
    attn_weights = self.attn(embed_hidden)[:, :seq_len]
    attn_weights = F.softmax(attn_weights, dim=1) # take only `seq_len` first items
    attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                             encoder_states.permute(1, 0, 2)).squeeze(1)

    output = torch.cat((decoder_embeds[0], attn_applied), 1)
    output = self.attn_combine(output).unsqueeze(0)

    return output
