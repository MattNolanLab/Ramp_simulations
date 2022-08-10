import json
import pandas as pd
import typing as t
from collections import defaultdict
from pathlib import Path


def get_behaviour_visual(run_path: Path):
  df = defaultdict(list)
  with open(run_path, 'r') as f:
    data = json.load(f)
    for timestep in range(len(data['location'])):
      df['timestep'].append(timestep)
      df['action'].append(data['action'][timestep])
      df['location'].append(data['location'][timestep])
      df['trial'].append(data['trial'][timestep])
      df['inter_trial'].append(data['trial'][timestep])
      df['trial_type'].append('nb')
      
  return pd.DataFrame.from_dict(df)
