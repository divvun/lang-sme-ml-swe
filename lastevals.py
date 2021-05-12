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
sami_tokenized_test, swedish_tokenized_test = tokenizer_sp(sami_model, [sami_sent[i] for i in test_indices if i in range(len(sami_sent))], swedish_model, [swedish_sent[i] for i in test_indices if i in range(len(swedish_sent))])
# [6] END

# [7] START
test_src = torch.stack([torch.LongTensor(sent) for sent in sami_tokenized_test])
test_tgts = torch.stack([torch.LongTensor(sent) for sent in swedish_tokenized_test])
# [7] END

# [8] START
test_short = []
for i in test_indices:
    if len(sami_sent[i].split()) <= 20:
        test_short.append(i)
# [8] END

# [9] START
sami_short = [sami_sent[i] for i in test_short]

swedish_short = [swedish_sent[i] for i in test_short]
# [9] END

# [10] START
len(test_short)
# [10] END

# [11] START
len(test_indices)
# [11] END

# [12] START
sami_tokenized, swedish_tokenized = tokenizer_sp(sami_model, sami_short, swedish_model, swedish_short)
# [12] END

# [13] START
len(sami_tokenized)
# [13] END

# [14] START
sami_tokenized = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in range(len(sami_tokenized))])
# [14] END

# [15] START
swedish_tokenized = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in range(len(swedish_tokenized))])
# [15] END

# [16] START
sami_tokenized.shape
# [16] END

# [17] START
max_len = 150
# [17] END

# [18] START
# Training hyperparameters
num_epochs = 500
learning_rate = 3e-4
batch_size = 64
# [18] END

# [19] START
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
# [19] END

# [20] START
load_model = True
# [20] END

# [25] START
# Synth SWE
# [25] END

# [26] START
model_path = "uni_joint_2layer_gelu_synth.pth.tar"
# [26] END

# [27] START
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
# [27] END

# [28] START
sp.Load(swedish_model)
criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
optimizer = optim.Adam(model_synth_swe.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
# [28] END

# [29] START
if load_model == True:
    checkpoint = torch.load(model_path, map_location='cpu')
    model_synth_swe.load_state_dict(checkpoint['state_dict'])           
    optimizer.load_state_dict(checkpoint['optimizer'])
model_synth_swe.to(device)
# [29] END

# [34] START
print("all sentences:")
# [34] END

# [35] START
print(get_scores(test_src, test_tgts, model_synth_swe, device, sami_model, swedish_model, "greedy"))
# [35] END

# [36] START
print("short sentences:")
# [36] END

# [37] START
print(get_scores(sami_tokenized, swedish_tokenized, model_synth_swe, device, sami_model, swedish_model, "greedy"))
# [37] END

# [38] START

# [38] END

# [39] START
## Synth SME
# [39] END

# [40] START
model_path = "uni_joint_2layer_gelu_synth_sme.pth.tar"
# [40] END

# [41] START
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
# [41] END

# [42] START
sp.Load(swedish_model)
criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
optimizer = optim.Adam(model_synth_sme.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
# [42] END

# [43] START
if load_model == True:
    checkpoint = torch.load(model_path, map_location='cpu')
    model_synth_sme.load_state_dict(checkpoint['state_dict'])           
    optimizer.load_state_dict(checkpoint['optimizer'])
model_synth_sme.to(device)
# [43] END

# [48] START
print(get_scores(sami_tokenized, swedish_tokenized, model_synth_swe, device, sami_model, swedish_model, "greedy"))
# [48] END

# [49] START

# [49] END

