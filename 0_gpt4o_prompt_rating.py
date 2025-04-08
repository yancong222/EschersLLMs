# -*- coding: utf-8 -*-
"""
Language models demonstrate the good-enough processing seen in humans
Cong and Rayz, CogSci 2025
"""

import pandas as pd
import numpy as np
import re
import math
import os
import string
import shutil, sys, glob

data = ('/content/path/')

"""# set up exp stimuli"""

df = pd.read_csv(data + 'temp.csv')
df

"""# openai-prompt set up"""

# !pip install openai # run this if you do not have openai installed
from google.colab import userdata
OPENAI_API_KEY = userdata.get('OPENAI_API_PROJECT_KEY')

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# models: https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a participant of a behavioral experiment."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

temp = completion.choices[0].message
print(temp)

print(temp.content)

"""# run it over the file - interpretation"""

prompt = "Promt: If an interpretation exists for the following statement, provide a concise 1-sentence interpretation; otherwise, output \"no interpretation\": "

print(prompt)

df['gpt4omini_interpret'] = ''

for i in range(len(df)):
    if type(df['scenario'][i]) != float:
      statement = 'Context: ' + df['scenario'][i] + ' ' + 'Statement: ' + df['possible_statements'][i]

      completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
          {"role": "system", "content": "You are a participant of a behavioral experiment."},
          {
              "role": "user",
              "content": prompt + statement
          }])
      temp = completion.choices[0].message.content
      df['gpt4omini_interpret'][i] = temp.replace(' \n', '')
      df.to_csv(data + 'escher_llms_stimuli_gpt4omini_promt.csv')

    else:
      statement = df['possible_statements'][i]
      completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
          {"role": "system", "content": "You are a participant of a behavioral experiment."},
          {
              "role": "user",
              "content": prompt + statement
          }])
      temp = completion.choices[0].message.content
      df['gpt4omini_interpret'][i] = temp.replace(' \n', '')
      df.to_csv(data + 'temp.csv')

    if i % 100 == 0:
      print('finished line: ', i)

df

"""# openai-prompt rating"""

prompt = "Rate how acceptable you find the sentence on a scale from 1 to 7, where 1 means \"not acceptable\" and 7 means \"completely acceptable\". Output a number 1 to 7 without explaining. "

print(prompt)

df['gpt4omini_rating'] = ''

for i in range(len(df)):
  if i > 1: # have a few trials here to experiment then run the whole dataset
    if type(df['scenario'][i]) != float:
      statement = 'Context: ' + df['scenario'][i] + ' ' + 'Sentence: ' + df['possible_statements'][i]

      completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
          {"role": "system",
           "content": "You are a participant of a behavioral experiment."},
          {
              "role": "user",
              "content": prompt + statement
          }])
      temp = completion.choices[0].message.content
      df['gpt4omini_rating'][i] = temp.replace(' \n', '')
      df.to_csv(data + 'escher_llms_stimuli_gpt4omini_promt.csv')

    else:
      statement = 'Sentence: ' + df['possible_statements'][i]
      completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
          {"role": "system", 
           "content": "You are a participant of a behavioral experiment."},
          {
              "role": "user",
              "content": prompt + statement
              #"seed" = seed # set a seed to reproduce
          }])
      temp = completion.choices[0].message.content
      df['gpt4omini_rating'][i] = temp.replace(' \n', '')
      df.to_csv(data + 'temp.csv')

    if i % 100 == 0:
      print('finished line: ', i)

df['gpt4omini_rating'] = pd.to_numeric(df['gpt4omini_rating'], errors='coerce')
stats = df.groupby(['experiment', 'condition'])['gpt4omini_rating'].agg(['mean', 'std', 'min', 'max', 'count'])

stats