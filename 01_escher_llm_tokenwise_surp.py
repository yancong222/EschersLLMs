# -*- coding: utf-8 -*-
# Date: May 2024 - Jan 2025
# Authors: Cong and Rayz

# Prep load libraries


import pandas as pd
import os

os.environ['HF_HOME'] = '/todo/'

df = pd.read_csv('todo.csv')
df

"""# utility functions"""

def tokenwise_surp(response, model):
  score = model.token_score([response], surprisal = True, base_two = True)
  return score[0]

"""# Minicons more opensourse LLMs load"""

from minicons import scorer
import torch

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(available_gpus)
print(gpu_names)

import gc
torch.cuda.empty_cache()
gc.collect()

"""## load models"""

from huggingface_hub import login
login("todo")

llama2 = scorer.IncrementalLMScorer('meta-llama/Llama-2-7b-hf', 'cuda')
falcon = scorer.IncrementalLMScorer('tiiuae/falcon-7b', 'cuda')
mpt = scorer.IncrementalLMScorer('mosaicml/mpt-7b', 'cuda')
mistral = scorer.IncrementalLMScorer('mistralai/Mistral-7B-v0.1', 'cuda')
qwen = scorer.IncrementalLMScorer('Qwen/Qwen1.5-MoE-A2.7B', 'cuda')
llama213b = scorer.IncrementalLMScorer('meta-llama/Llama-2-13b-hf', 'cuda')

"""# Run LLMs on the whole dataset"""

df['tokenwise_surp_MPT'] = df['possible_statements'].apply(lambda x: tokenwise_surp(x, mpt))
df['tokenwise_surp_Mistral'] = df['possible_statements'].apply(lambda x: tokenwise_surp(x, mistral))
df['tokenwise_surp_Llama'] = df['possible_statements'].apply(lambda x: tokenwise_surp(x, llama2))
df['tokenwise_surp_Falcon'] = df['possible_statements'].apply(lambda x: tokenwise_surp(x, falcon))
df['tokenwise_surp_qwen'] = df['possible_statements'].apply(lambda x: tokenwise_surp(x, qwen))
df['tokenwise_surp_llama213b'] = df['possible_statements'].apply(lambda x: tokenwise_surp(x, llama213b))

df.to_csv('todo.csv')




