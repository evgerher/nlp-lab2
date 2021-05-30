from argparse import ArgumentParser
import logging
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from src.logger import setup_logger
from src.models.experiment_setup import ExperimentSetup
from src.models.seq2seq import Seq2Seq
from src.train.loop import train_model
from src.train.train_setup import TrainSetup

logger = logging.getLogger('runner')


def eval_callback(args):
  logger.info('Selected EVALUATION callback')

  pass

  logger.info('Finished EVALUATION callback')


def train_callback(args):
  logger.info('Selected TRAIN callback')
  experiment_name = args.experiment_name
  experiment_path = args.architecture_path
  learning_path = args.learning_path
  experiment_setup = ExperimentSetup.load_yaml(experiment_path)
  train_setup = TrainSetup.load_yaml(learning_path)

  time_specific = datetime.now()
  time_str = time_specific.strftime('%H_%M_%S')
  writer = SummaryWriter(f'experiment_{experiment_name}_{time_str}')

  seq2seq, en_vocab, ru_vocab = Seq2Seq.build(experiment_setup, experiment_name)
  seq2seq = seq2seq.to(seq2seq.device)
  seq2seq = train_model(seq2seq, train_setup, experiment_setup, en_vocab, ru_vocab, writer)
  logger.info('Finished TRAIN callback')

def parse_arguments():
  parser = ArgumentParser('CLI tool to train and evaluate nlp models for NMT task')
  subparsers = parser.add_subparsers(help='Choose command to use')

  train_cli = subparsers.add_parser('train', help='Train NMT model')
  eval_cli = subparsers.add_parser('eval', help='Evaluate NMT model')

  train_cli.add_argument('--experiment_name',
                         required=True,
                         type=str,
                         help='Provide a name for your experiment')
  train_cli.add_argument('--architecture_path',
                         required=True,
                         type=str,
                         help='Path to yaml file with seq2seq architecture')
  train_cli.add_argument('--learning_path',
                         required=True,
                         type=str,
                         help='Path to yaml file with train configs')


  train_cli.set_defaults(callback=train_callback)
  eval_cli.set_defaults(callback=eval_callback)

  return parser.parse_args()


def main():
  args = parse_arguments()
  args.callback(args)


if __name__ == '__main__':
  setup_logger()
  main()
