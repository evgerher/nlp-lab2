from typing import Optional

import torch

from src.train.train_setup import OptimizerSetup, SchedulerSetup


def resolve_optimizer(params, setup: OptimizerSetup):
  if setup.name == 'adam':
    return torch.optim.Adam(params, **setup.kwargs)
  elif setup.name == 'rmsprop':
    return torch.optim.RMSprop(params, **setup.kwargs)
  elif setup.name == 'adamw':
    return torch.optim.AdamW(params, **setup.kwargs)
  else:
    raise NotImplementedError()

def resolve_scheduler(optim, setup: SchedulerSetup):
  if setup.name == 'plateau':
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **setup.kwargs)
  else:
    return None
