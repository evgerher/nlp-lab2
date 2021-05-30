import torch
from torch import nn

from src.models.experiment_setup import EmbeddingSetup
from src.models.static import Encoder


class ConvModel1(Encoder):
  def __init__(self, emb_setup: EmbeddingSetup):
    super(ConvModel1, self).__init__()
    self.input_size = emb_setup.embedding_size

  def forward(self, embedding, hidden):
    pass
