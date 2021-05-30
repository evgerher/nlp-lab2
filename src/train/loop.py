from datetime import datetime
from typing import List
import os

import datasets
import torch
from datasets import load_dataset, load_metric # https://github.com/huggingface/datasets
import spacy
from tqdm import trange
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from src.models.seq2seq import Seq2Seq
from .train_setup import TrainSetup
from .resolver import resolve_optimizer, resolve_scheduler
from ..models.embed.consts import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from ..models.embed.resolver import *
from ..models.experiment_setup import EmbeddingSetup, ExperimentSetup


class MyTokenizer:
  def __init__(self, words: VOCAB, spacy_tokenizer, lemmatize: bool, max_length: int):
    self.i2w = {}
    self.w2i = {}
    self.unk_token = UNK_TOKEN
    self.bos_token = BOS_TOKEN
    self.eos_token = EOS_TOKEN
    self.pad_token = PAD_TOKEN
    self.tokenizer = spacy_tokenizer
    self.is_lemmatizer = lemmatize
    self.max_length = max_length - 2 # take into account BOS and EOS

    for idx, word in enumerate(words):
      self.i2w[idx] = word
      self.w2i[word] = idx

    self.unk_token_id = self.w2i[UNK_TOKEN]
    self.bos_token_id = self.w2i[BOS_TOKEN]
    self.eos_token_id = self.w2i[EOS_TOKEN]
    self.pad_token_id = self.w2i[PAD_TOKEN]

    self.special_token_ids = set(self.w2i[token] for token in SPECIAL_TOKENS)

  def pad(self, examples, return_tensors='pt'):
    max_length = max(map(len, examples))
    padded_examples = [token_seq + [self.pad_token_id] * (max_length - len(token_seq)) for token_seq in examples]
    tensor = torch.tensor(padded_examples, dtype=torch.long)
    return tensor # todo: where I add BOS and EOS?

  def get_tokens(self, seq: str):
    words = self.tokenizer(seq)[:self.max_length]
    if self.is_lemmatizer:
      lemmas = [word.lemma_ for word in words]
      return lemmas
    else:
      words = [str(w) for w in words]
    return words

  def encode_batch(self, seqs: List[str]) -> List[List[int]]:
    token_batch = []
    for seq in seqs:
      seq = self.get_tokens(seq)
      tokens = [self.w2i.get(w, self.unk_token_id) for w in seq]
      token_batch.append(tokens)
    return token_batch

  def encode(self, seq: str, add_bos_eos=True) -> List[int]:
    seq = self.get_tokens(seq)
    tokens = [self.w2i.get(w, self.unk_token_id) for w in seq] # replace with unknown token if not in vocabulary
    if add_bos_eos:
      tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
    return tokens

  def encode_words(self, seq: List[str], max_length: int) -> List[int]:
    tokens = [self.w2i.get(w, self.unk_token_id) for w in seq][:max_length] # todo: think by yourself!!! include or not eos/bos
    return tokens

  def decode(self, tokens: List[int], disable_special_tokens: bool = True) -> List[str]:
    if disable_special_tokens:
      tokens = [x for x in tokens if x not in self.special_token_ids]
    words = [self.i2w.get(t, self.unk_token) for t in tokens]
    return words


def prepare_actors(seq2seq: Seq2Seq, train_params: TrainSetup):
  optimizer = resolve_optimizer(seq2seq.parameters(), train_params.optimizer)
  scheduler = resolve_scheduler(optimizer, train_params.scheduler)
  criterion = torch.nn.NLLLoss()
  return optimizer, scheduler, criterion


def lemmatize_tokenize_dataset(dataset, tokenizer: MyTokenizer, language: str, input_columns: List[str]):
  def tokenization(example):
    tokens = tokenizer.encode(example[language], add_bos_eos=True)
    return {f'{language}_tokens': tokens}
  ds = dataset.map(tokenization, batched=False, input_columns=input_columns)
  return ds


def prepare_dataset(en_vocab: VOCAB,
                    ru_vocab: VOCAB,
                    en_emb_setup: EmbeddingSetup,
                    ru_emb_setup: EmbeddingSetup):

  ds_name = f'opus100-{en_emb_setup.token_type}-{ru_emb_setup.token_type}'
  dataset_present = os.path.exists(ds_name)

    # https://github.com/huggingface/datasets
  if not dataset_present:
    dataset = load_dataset("opus100", "en-ru")
    dataset['train'] = dataset['test'] # todo: make train smaller
    del dataset['test']
  if en_emb_setup.token_type == 'word':
    en_nlp = spacy.load("en_core_web_sm")
    en_tokenizer = MyTokenizer(en_vocab, en_nlp, True, en_emb_setup.max_length)
    if not dataset_present:
      en_dataset = lemmatize_tokenize_dataset(dataset, en_tokenizer, 'en', input_columns=['translation'])


  elif en_emb_setup.token_type == 'bpe': # vocabulary is not used, custom tokenizer is also not possible
    en_bpe_tokenizer = AutoTokenizer.from_pretrained(en_emb_setup.name) # expecting model name
    en_bpe_tokenizer.add_special_tokens({'bos_token': BOS_TOKEN,
                                         'eos_token': EOS_TOKEN,
                                         'pad_token': PAD_TOKEN,
                                         'unk_token': UNK_TOKEN})
    en_tokenizer = en_bpe_tokenizer
    if not dataset_present:
      dataset = dataset.map(en_bpe_tokenizer.__call__, input_columns='translation', batched=True)
  else:
    raise NotImplementedError()

  if ru_emb_setup.token_type == 'word':
    ru_nlp = spacy.load("ru_core_news_sm")
    ru_tokenizer = MyTokenizer(ru_vocab, ru_nlp, False, ru_emb_setup.max_length)
    if not dataset_present:
      ru_dataset = lemmatize_tokenize_dataset(dataset, ru_tokenizer, 'ru', input_columns=['translation'])
      en_dataset['train'] = en_dataset['train'].add_column('ru_tokens', ru_dataset['train']['ru_tokens'])
      en_dataset['validation'] = en_dataset['validation'].add_column('ru_tokens', ru_dataset['validation']['ru_tokens'])
      dataset = en_dataset


  elif ru_emb_setup.token_type == 'bpe': # vocabulary is not used, custom tokenizer is also not possible
    # works only with "DeepPavlov/rubert-base-cased" ?
    ru_bpe_tokenizer = AutoTokenizer.from_pretrained(ru_emb_setup.name)
    ru_bpe_tokenizer.add_special_tokens({'bos_token': BOS_TOKEN,
                                         'eos_token': EOS_TOKEN,
                                         'pad_token': PAD_TOKEN,
                                         'unk_token': UNK_TOKEN})
    ru_tokenizer = ru_bpe_tokenizer
    if not dataset_present:
      dataset = dataset.map(ru_bpe_tokenizer.__call__, input_columns='translation', batched=True)
  else:
    raise NotImplementedError()
  if not dataset_present:
    dataset.save_to_disk(ds_name)
  if dataset_present:
    dataset = datasets.load_from_disk(ds_name)

  return dataset, en_tokenizer, ru_tokenizer
  # todo: think about collate with length ?


class Collator:
  def __init__(self, tokenizer_ru, tokenizer_en):
    self.tokenizer_ru = tokenizer_ru
    self.tokenizer_en = tokenizer_en

  def collate_fn(self, examples):
    en_tokens = []
    ru_tokens = []
    for example in examples:
      en_tokens.append(example['en_tokens'])
      ru_tokens.append(example['ru_tokens'])

    en_pads = self.tokenizer_en.pad(en_tokens, return_tensors='pt')
    ru_pads = self.tokenizer_ru.pad(ru_tokens, return_tensors='pt')

    return (en_pads, ru_pads)

def prepare_dataloaders(dataset, tokenizer_en, tokenizer_ru, batch_size: int, num_workers: int):
  collator = Collator(tokenizer_ru, tokenizer_en)
  train_dataloader = torch.utils.data.DataLoader(dataset['train'], collate_fn=collator.collate_fn, batch_size=batch_size, num_workers=num_workers)
  val_dataloader = torch.utils.data.DataLoader(dataset['validation'], collate_fn=collator.collate_fn, batch_size=batch_size, num_workers=num_workers)
  # next(iter(train_dataloader))
  # next(iter(val_dataloader))
  return train_dataloader, val_dataloader


def train_model(seq2seq: Seq2Seq,
                train_setup: TrainSetup,
                experiment_setup: ExperimentSetup,
                en_vocab: VOCAB,
                ru_vocab: VOCAB,
                writer: SummaryWriter):
  logger.info('Start initializing pipelines')
  optimizer, scheduler, criterion = prepare_actors(seq2seq, train_setup)
  dataset, en_tokenizer, ru_tokenizer = prepare_dataset(en_vocab,
                                                        ru_vocab,
                                                        experiment_setup.encoder_embedding,
                                                        experiment_setup.decoder_embedding)
  train_dataloader, val_dataloader = prepare_dataloaders(dataset, en_tokenizer,
                                                         ru_tokenizer,
                                                         train_setup.batch_size,
                                                         train_setup.num_workers)
  logger.info('Initialized datasets and dataloaders')

  start_time = datetime.now()
  logger.info('Start training')
  seq2seq = train_pipeline(seq2seq,
                 optimizer,
                 scheduler,
                 criterion,
                 train_dataloader,
                 val_dataloader,
                 train_setup.epochs,
                 writer)
  logger.info('Finish training')
  end_time = datetime.now()
  total_time = (end_time - start_time).total_seconds() / 60.

  msg = 'Total train time (min): %.2f'.format(total_time)
  logger.info(msg)
  writer.add_scalar('TRAIN TIME (min)', total_time)

  start_time = datetime.now()
  logger.info('Start BLEU estimation')
  bleu = estimate_bleu(seq2seq, val_dataloader, ru_tokenizer.bos_token_id, ru_tokenizer.eos_token_id)
  logger.info('Model BLEU: %.2f', bleu)
  writer.add_scalar('Model BLEU', bleu)
  end_time = datetime.now()
  total_time = (end_time - start_time).total_seconds() / 60.
  msg = 'Total BLEU time (min): %.2f'.format(total_time)
  logger.info(msg)
  writer.add_scalar('BLEU TIME (min)', total_time)

  return seq2seq



def train_pipeline(seq2seq: Seq2Seq,
                   optimizer,
                   scheduler,
                   criterion,
                   train_dataloader,
                   val_dataloader,
                   epochs: int,
                   writer: SummaryWriter):

  best_val_loss = float('inf')
  for epoch in trange(epochs):
    seq2seq.train()
    train_loss = 0.0
    for idx, (en_tokens, ru_tokens) in enumerate(train_dataloader, 1):
      en_tokens, ru_tokens = en_tokens.to(seq2seq.device), ru_tokens.to(seq2seq.device)
      loss = seq2seq.train_batch(en_tokens, ru_tokens, optimizer, criterion, scheduler)
      train_loss += loss
    else:
      train_loss /= idx
      logger.info('Epoch [%d], train loss: %f', epoch, train_loss)
      if writer is not None:
        writer.add_scalar('Train loss', train_loss, epoch)

      with torch.no_grad():
        seq2seq.eval()
        eval_loss = 0.0
        for idx, batch in enumerate(val_dataloader, 1):
          en_tokens, ru_tokens = batch['en_tokens'], batch['ru_tokens']
          loss = seq2seq.eval_batch(en_tokens, ru_tokens, criterion)
          eval_loss += loss
        eval_loss /= idx

        logger.info('Epoch [%d], val loss: %.4f', epoch, eval_loss)
        if writer is not None:
          writer.add_scalar('Val loss', eval_loss, epoch)
        if eval_loss < best_val_loss:
          logger.info('New best val loss: %.4f', epoch, eval_loss)
          best_val_loss = eval_loss
          seq2seq.save('best-model') # todo: me!!!!!
  return seq2seq


def estimate_bleu(seq2seq: Seq2Seq,
                  val_dataset,
                  bos_token_id: int,
                  eos_token_id: int) -> float:
  # perplexity, BLEU (benchmark),
  bleu_metric = load_metric('bleu')
  with torch.no_grad():
    seq2seq.eval()
    for example in val_dataset:
      en_tokens, ru_tokens = example['en_tokens'], example['ru_tokens']
      max_len = ru_tokens.shape[1] + 10
      predictions = seq2seq.predict(en_tokens, bos_token_id, eos_token_id, max_len) # todo: me
      bleu_metric.add_batch(predictions, ru_tokens)
  score = bleu_metric.compute()
  return score


def final_estimation():
  # number of epochs used during training
  # примеры удачных и неудачных примеров
  # идея эксперимента, иллюстрация процесса на графиках
  # оценка оверфита
  # inference speed vs batch of 32 items
  pass

