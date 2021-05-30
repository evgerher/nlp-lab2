from dataclasses import dataclass, field
from typing import Optional, Union
import logging

import yaml
from marshmallow_dataclass import class_schema

logger = logging.getLogger('runner')

@dataclass
class EmbeddingSetup:  # transformers ignore pretrained embeddings
  embedding_size: int
  vocabulary_size: Optional[int]  # not including SPECIAL_TOKENS
  vocabulary_file: Optional[str]
  name: Optional[str]  # take pretrained if name matches
  language: str  # ru or en
  token_type: str # word or bpe # bpe is only for transformers
  max_length: int


class ModelArchitecture:
  @classmethod
  def build(cls, kwargs: dict):
    if 'cell' in kwargs:
      return RNN_Architecture(**kwargs)
    elif 'conv_name' in kwargs:
      return ConvArchitecture(**kwargs)
    elif 'model_type' in kwargs:
      return TransformerArchitecture(**kwargs)


@dataclass
class RNN_Architecture(ModelArchitecture):
  layers: int
  layer_dropout: Optional[float]
  dropout: Optional[float]
  cell: str # GRU, LSTM, RNN
  hidden_size: int
  bidirectional: bool = field(default=False)


@dataclass
class AttentionSetup(ModelArchitecture):
  name: str
  max_length: Optional[int] # required for TutorialAttention

@dataclass
class ConvArchitecture(ModelArchitecture):
  conv_name: str # only two options: conv-1, conv-2 # todo: implement those options later


@dataclass
class TransformerArchitecture(ModelArchitecture):
  model_type: str # gpt-2, bert, transformer, etc (from trnasformers library)
  model_size: str
  pretrained: bool = field(default=True)



@dataclass
class ExperimentSetup:
  attention: AttentionSetup
  encoder_embedding: EmbeddingSetup
  decoder_embedding: EmbeddingSetup
  encoder: dict # transformers ignore pretrained embeddings
  decoder: dict # transformers ignore pretrained embeddings

  @classmethod
  def load_yaml(cls, path: str) -> 'TrainSetup':
    with open(path, "r") as fread:
      schema = ExperimentSetupSchema()
      config = schema.load(yaml.safe_load(fread))
      logger.info('Loaded experiment config from: %s', path)
      return config

ExperimentSetupSchema = class_schema(ExperimentSetup)
MODEL_ARCHITECTURE = ModelArchitecture

