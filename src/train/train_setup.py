import logging
from dataclasses import dataclass
from typing import Optional
import logging

import yaml
from marshmallow_dataclass import class_schema

logger = logging.getLogger('runner')


@dataclass
class SchedulerSetup:
  name: str
  kwargs: Optional[dict]

@dataclass
class OptimizerSetup:
  name: str
  kwargs: Optional[dict]



@dataclass
class TrainSetup:
  optimizer: OptimizerSetup
  scheduler: Optional[SchedulerSetup]
  epochs: int
  batch_size: int

  @classmethod
  def load_yaml(cls, path: str) -> 'TrainSetup':
    with open(path, "r") as fread:
      schema = TrainSetupSchema()
      config = schema.load(yaml.safe_load(fread))
      logger.info('Loaded train config from: %s', path)
      return config

TrainSetupSchema = class_schema(TrainSetup)
