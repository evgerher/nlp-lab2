
import logging
from typing import Tuple

from torch import nn

from src.models.experiment_setup import EmbeddingSetup
from .consts import SPECIAL_TOKENS, VOCAB
from .glove import resolve_glove_en

logger = logging.getLogger('runner')


# todo: where do I handle vocabulary?
# todo: add special tokens to vocabulary list


def resolve_embedding(emb_setup: EmbeddingSetup) -> Tuple[VOCAB, nn.Embedding]:
  emb_setup.vocabulary_size = emb_setup.vocabulary_size + len(SPECIAL_TOKENS)
  padding_idx = emb_setup.vocabulary_size - 1

  if emb_setup.name == 'glove' and emb_setup.language == 'en':
    logger.info('Loading GLOVE 300d, EN embedding')
    vocabulary, embedder = resolve_glove_en(emb_setup, padding_idx)
  else:
    logger.info('Loading default embedding')
    embedder = nn.Embedding(emb_setup.vocabulary_size,
                            emb_setup.embedding_size,
                            padding_idx=padding_idx)
    vocabulary = None
  return vocabulary, embedder


def load_vocabulary(path: str) -> VOCAB:
  with open(path, 'r') as fr:
    words = [w for w in fr.read().split('\n') if w] # word per line
    return words + SPECIAL_TOKENS
