import logging
from typing import Optional, Tuple
import zipfile
import os
import requests

import numpy as np
import torch
from torch import nn

from .consts import SPECIAL_TOKENS, VOCAB
from src.models.experiment_setup import EmbeddingSetup

logger = logging.getLogger('runner')

def resolve_glove_en(emb_setup: EmbeddingSetup, padding_idx: Optional[int] = None) -> Tuple[VOCAB, nn.Embedding]:
  glove_local_pt_path = 'glove_embeds.pt'
  vocab_local_path = 'glove_words.txt'
  vocab_size = emb_setup.vocabulary_size
  emb_size = 300

  if not os.path.isfile(glove_local_pt_path):
    link = 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'
    fname = 'glove.6B.zip'
    if not os.path.isfile(fname):
      logger.info('Downloading [%s]', link)
      download_file(link, fname)

    logger.info('Open zipfile with embeds')
    with zipfile.ZipFile(fname) as zip_file:
      with zip_file.open('glove.6B.300d.txt', 'r') as emb_file:
        contents: str = emb_file.read().decode('utf-8')

    words, embeds = [], []
    for row in contents.split('\n'):
      items = row.split(' ')
      word, embed = items[0], items[1:]
      if len(word) > 0:
        words.append(word)
        embeds.append(embed)

    assert vocab_size == len(words) + len(SPECIAL_TOKENS), 'Invalid number of words after unpackaging'
    assert len(embeds[0]) == emb_size, f'Expected embedding size to be {emb_size}; found {len(embeds[0])}'

    embeds = np.array(embeds, dtype=np.float32)
    word_embeds = torch.from_numpy(embeds).float()
    torch.save(word_embeds, glove_local_pt_path)
    with open(vocab_local_path, 'w') as fo:
      for word in words[:-1]:
        fo.write(word)
        fo.write('\n')
      fo.write(words[-1])
  else:
    word_embeds = torch.load(glove_local_pt_path)
    with open(vocab_local_path, 'r') as fo:
      words = fo.read().split('\n')
  weights = torch.rand(vocab_size, emb_size)
  weights[:len(word_embeds)] = word_embeds  # others are initialized randomly

  embedder = nn.Embedding.from_pretrained(weights, padding_idx=padding_idx)
  return words + SPECIAL_TOKENS, embedder


def download_file(link: str, path: str):
  response = requests.get(link, allow_redirects=True)
  with open(path, 'wb') as fw:
    fw.write(response.content)
