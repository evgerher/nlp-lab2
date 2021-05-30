import torch
from torch import nn
import torch.nn.functional as F

from src.models.experiment_setup import RNN_Architecture
from src.models.static import Encoder, Decoder
from src.models.attention import Attention


def resolve_rnn(input_dim: int, model_setup: RNN_Architecture) -> nn.Module:
  cell_name = model_setup.cell
  if cell_name == 'GRU':
    constructor = nn.GRU
  elif cell_name == 'LSTM':
    constructor = nn.LSTM
  elif cell_name == 'RNN':
    constructor = nn.RNN
  else:
    raise NotImplementedError()

  return constructor(
      input_size=input_dim,
      hidden_size=model_setup.hidden_size,
      dropout=model_setup.dropout,
      num_layers=model_setup.layers,
      bidirectional=model_setup.bidirectional
    )


class RNN_ModelEncoder(Encoder):
  def __init__(self, model_setup: RNN_Architecture, embedding: nn.Embedding):
    super(RNN_ModelEncoder, self).__init__()
    self.input_dim = embedding.embedding_dim
    self.hid_dim = model_setup.hidden_size
    self.nlayers = model_setup.layers
    self.bidirectional = model_setup.bidirectional
    self.cell_type = model_setup.cell

    self.dropout = nn.Dropout(model_setup.dropout)
    self.rnn = resolve_rnn(self.input_dim, model_setup)
    self.embedding = embedding

  def forward(self, input, hidden, *args):
    embeds = self.embedding(input)
    if hidden is None and self.cell_type == 'LSTM':
      hidden = (None, None)
    output, new_hidden = self.rnn(embeds, hidden)
    return output, new_hidden


class RNN_ModelDecoder(RNN_ModelEncoder, Decoder):
  def __init__(self, model_setup: RNN_Architecture, embedding: nn.Embedding, attention: Attention, out_classes: int):
    super().__init__(model_setup, embedding)
    self.attention = attention
    self.out = nn.Linear(model_setup.hidden_size, out_classes)

  def forward(self, input, hidden, encoder_outputs):
    embeds = self.embedding(input).unsqueeze(0)
    embeds = self.dropout(embeds)
    if self.attention:
      rnn_input = self.attention(embeds, hidden, encoder_outputs)
    else:
      rnn_input = embeds
    rnn_input = F.relu(rnn_input)
    output, hidden = self.rnn(rnn_input, hidden)
    output = self.out(output)[0]
    return output, hidden
