from abc import ABC

from torch import nn

from src.models.experiment_setup import MODEL_ARCHITECTURE, RNN_Architecture, ConvArchitecture, TransformerArchitecture

class Encoder(nn.Module, ABC):
  @classmethod
  def resolve_architecture(cls,
                           architecture: MODEL_ARCHITECTURE,
                           embedder: nn.Embedding):
    if type(architecture) is RNN_Architecture:
      from src.models.rnn_model import RNN_ModelEncoder
      return RNN_ModelEncoder(architecture, embedder)
    elif type(architecture) is ConvArchitecture:
      raise NotImplementedError()
    elif type(architecture) is TransformerArchitecture:
      raise NotImplementedError()


class Decoder(nn.Module, ABC):
  @classmethod
  def resolve_architecture(cls,
                           architecture: MODEL_ARCHITECTURE,
                           embedder: nn.Embedding,
                           attention: 'Attention',
                           out_classes: int):
    if type(architecture) is RNN_Architecture:
      from src.models.rnn_model import RNN_ModelDecoder
      return RNN_ModelDecoder(architecture, embedder, attention, out_classes)
    elif type(architecture) is ConvArchitecture:
      raise NotImplementedError()
    elif type(architecture) is TransformerArchitecture:
      raise NotImplementedError()