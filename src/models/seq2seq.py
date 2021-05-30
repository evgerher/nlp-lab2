from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import random

from src.models.attention import Attention
from src.models.embed import resolve_embedding, load_vocabulary, VOCAB
from src.models.experiment_setup import ExperimentSetup, ModelArchitecture
from src.models.static import Encoder, Decoder

random.seed(34)
torch.random.manual_seed(34)


class Seq2Seq(nn.Module):
  @classmethod
  def build(cls, experiment: ExperimentSetup) -> Tuple['Seq2Seq', VOCAB, VOCAB]:
    enc_emb_setup = experiment.encoder_embedding
    enc_vocabulary, enc_embedder = resolve_embedding(enc_emb_setup)
    if enc_vocabulary is None:
      enc_vocabulary = load_vocabulary(enc_emb_setup.vocabulary_file)

    dec_emb_setup = experiment.decoder_embedding
    dec_vocabulary, dec_embedder = resolve_embedding(dec_emb_setup)
    if dec_vocabulary is None:
      dec_vocabulary = load_vocabulary(dec_emb_setup.vocabulary_file)
      dec_emb_setup.vocabulary_size = len(dec_vocabulary)

    encoder_architecture = ModelArchitecture.build(experiment.encoder)
    decoder_architecture = ModelArchitecture.build(experiment.decoder)
    attention = Attention.resolve(experiment.attention,
                                  encoder_architecture,
                                  decoder_architecture,
                                  enc_emb_setup,
                                  dec_emb_setup)
    encoder = Encoder.resolve_architecture(encoder_architecture, enc_embedder)
    decoder = Decoder.resolve_architecture(decoder_architecture, dec_embedder, attention, out_classes=len(dec_vocabulary))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return Seq2Seq(encoder, decoder, device), enc_vocabulary, dec_vocabulary


  def __init__(self, encoder: Encoder, decoder: Decoder, device):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device

    assert encoder.hid_dim == decoder.hid_dim, \
      "Hidden dimensions of encoder and decoder must be equal!"
    # assert encoder.n_layers == decoder.n_layers, \
    #   "Encoder and decoder must have equal number of layers!"

  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    src = src.T
    trg = trg.T
    # src = [src sent len, batch size]
    # trg = [trg sent len, batch size]
    # teacher_forcing_ratio is probability to use teacher forcing
    # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

    # Again, now batch is the first dimention instead of zero
    batch_size = trg.shape[1]
    max_len = trg.shape[0] # todo: look precisely here
    trg_vocab_size = self.decoder.embedding.num_embeddings # todo: what ?

    # tensor to store decoder outputs
    outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    encoder_output_states, encoder_hidden = self.encoder(src, None) # encoder_hidden can be pair

    # first input to the decoder is the <sos> tokens
    input = trg[0, :]
    decoder_hidden = encoder_hidden[:2]
    for t in range(1, max_len):
      # todo: i am not expecting softmax or log_softmax
      output, decoder_hidden = self.decoder(input, decoder_hidden, encoder_output_states)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.max(1)[1]
      input = (trg[t] if teacher_force else top1)

    return outputs

  def predict(self, src, bos_token_id, eos_token_id, max_len: int):
    batch_size = src.shape[1] # should be 1
    encoder_output_states, encoder_hidden = self.encoder(src)  # encoder_hidden can be pair

    input = torch.tensor([bos_token_id] * batch_size, dtype=torch.long, device=self.device).view(batch_size, 1)
    decoder_hidden = encoder_hidden

    seq = [bos_token_id]
    local_len = 1
    last_token = bos_token_id
    while last_token != eos_token_id or local_len <= max_len:
      output, decoder_hidden = self.decoder(input, encoder_output_states, decoder_hidden)
      input = output.max(1)[1]
      token_id = input.item()
      seq += [token_id]
      last_token = token_id

    return seq


  def train_batch(self, input_tensor, target_tensor, optim, criterion, scheduler):
    # todo: use scheduler
    batch_size = len(target_tensor)
    optim.zero_grad()
    outputs = self.forward(input_tensor, target_tensor)
    loss: torch.Tensor = 0.0
    for idx, output in enumerate(outputs):
      log_softmax = F.log_softmax(output, dim=1)
      loss += criterion(log_softmax, target_tensor[:, idx])

    loss.backward()
    optim.step()
    return loss.item() / batch_size

  def eval_batch(self, input_tensor, target_tensor, criterion):
    batch_size = len(target_tensor)
    outputs = self.forward(input_tensor, target_tensor)
    loss = 0.0
    for idx, output in enumerate(outputs):
      log_softmax = F.log_softmax(output)
      loss += criterion(log_softmax, target_tensor[idx])
    return loss.item() / batch_size
