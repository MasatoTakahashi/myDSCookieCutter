from dataclasses import dataclass
from pathlib import Path


@dataclass
class Parameters:
  path_exec_root: str = Path('./')
  path_input: str = Path('./input')
  path_intermediate: str = Path('./intermediate')
  path_model: str = Path('./model')
  path_output: str = Path('./submit')
  path_log: str = Path('./log')
