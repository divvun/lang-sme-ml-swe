# [0] START
#!pip install nltk
#!pip install sacrebleu
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#!pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#!pip install sentencepiece
import sentencepiece as spm
from utils import *
from collections import Counter
from transformer_model import Transformer as Trans_model
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import json
import numpy as np
# [0] END

# [1] START
device = torch.device("cuda")
# [1] END

# [2] START
sp = spm.SentencePieceProcessor()
# [2] END

# [3] START
sami_model = "combbig.model"
swedish_model = "combbig.model"
# [3] END

# [4] START
train_indices = []
val_indices = []
test_indices = []
with open('train_indices_big.txt') as fp:
    for line in fp:
        train_indices.append(int(line))
with open('val_indices_big.txt') as fp:
    for line in fp:
        val_indices.append(int(line))
with open('test_indices_big.txt') as fp:
    for line in fp:
        test_indices.append(int(line))
# [4] END

# [5] START
sami_sent, swedish_sent = read_data('corpora/smeswebig.tmx.gz')
# [5] END

# [6] START
len(test_indices)
# [6] END

# [7] START
sami_tokenized, swedish_tokenized = tokenizer_sp(sami_model, sami_sent, swedish_model, swedish_sent)
# [7] END

# [8] START
len(sami_tokenized)
# [8] END

# [9] START
train_src = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in train_indices if i in range(len(sami_tokenized))])
train_tgts = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in train_indices if i in range(len(sami_tokenized))])
# [9] END

# [10] START
len(test_src)
# [10] END

# [11] START
test_short = []
for i in test_indices:
    if len(sami_sent[i].split()) <= 20:
        test_short.append(i)
# [11] END

# [12] START
sami_short = [sami_sent[i] for i in test_short]

swedish_short = [swedish_sent[i] for i in test_short]
# [12] END

# [13] START
len(test_short)
# [13] END

# [14] START
len(test_indices)
# [14] END

# [15] START
sami_tokenized_short, swedish_tokenized_short = tokenizer_sp(sami_model, sami_short, swedish_model, swedish_short)
# [15] END

# [16] START
len(sami_tokenized_short)
# [16] END

# [17] START
sami_tokenized_short = torch.stack([torch.LongTensor(sami_tokenized_short[i]) for i in range(len(sami_tokenized_short))])
# [17] END

# [18] START
swedish_tokenized_short = torch.stack([torch.LongTensor(swedish_tokenized_short[i]) for i in range(len(swedish_tokenized_short))])
# [18] END

# [19] START
sami_tokenized_short.shape
# [19] END

# [20] START
max_len = 150
# [20] END

# [21] START
# Training hyperparameters
num_epochs = 500
learning_rate = 3e-4
batch_size = 64
# [21] END

# [22] START
# Model hyperparameters
src_vocab_size = 16000
trg_vocab_size = 16000
embedding_size = 512
num_heads = 8
num_encoder_layers = 2 #6
num_decoder_layers = 2 #6
dropout = 0.10
max_len = 150
forward_expansion = 2048
sp.Load(sami_model)
src_pad_idx = sp.pad_id()
# [22] END

# [23] START
load_model = True
# [23] END

# [29] START
model_path = "uni_joint_2layer_gelu_synth.pth.tar"
# [29] END

# [30] START
model_synth_swe = Trans_model(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
    'gelu'
)
# [30] END

# [31] START
sp.Load(swedish_model)
criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
optimizer = optim.Adam(model_synth_swe.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
# [31] END

# [32] START
if load_model == True:
    checkpoint = torch.load(model_path, map_location='cpu')
    model_synth_swe.load_state_dict(checkpoint['state_dict'])           
    optimizer.load_state_dict(checkpoint['optimizer'])
model_synth_swe.to(device)
# [32] END

# [37] START
print("all sentences:")
# [37] END

# [38] START
len(test_src)
# [38] END

# [39] START
print(get_scores(test_src, test_tgts, model_synth_swe, device, sami_model, swedish_model, "greedy"))
# [39] END

# [40] START
## Synth SME
# [40] END

# [41] START
model_path = "uni_joint_2layer_gelu_synth_sme.pth.tar"
# [41] END

# [42] START
model_synth_sme = Trans_model(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
    'gelu'
)
# [42] END

# [43] START
sp.Load(swedish_model)
criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
optimizer = optim.Adam(model_synth_sme.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
# [43] END

# [44] START
if load_model == True:
    checkpoint = torch.load(model_path, map_location='cpu')
    model_synth_sme.load_state_dict(checkpoint['state_dict'])           
    optimizer.load_state_dict(checkpoint['optimizer'])
model_synth_sme.to(device)
# [44] END

# [49] START
print(get_scores(sami_tokenized_short, swedish_tokenized_short, model_synth_sme, device, sami_model, swedish_model, "greedy"))
# [49] END

