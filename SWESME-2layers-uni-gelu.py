# [0] START
# !pip install nltk
# !pip install sacrebleu
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
# !pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# !pip install sentencepiece
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
device = torch.device("cuda:0")
# [1] END

# [2] START
sami_model = "combbig.model"
swedish_model = "combbig.model"
# [2] END

# [3] START
model_path = "SWESME-backtranslate-uni_joint_2layer_gelu.pth.tar"
# [3] END

# [4] START
max_len = 150
# [4] END

# [5] START
sami_sent, swedish_sent = read_data('corpora/smeswebig.tmx')
# [5] END

# [6] START
len(sami_sent)
# [6] END

# [7] START
#spm.SentencePieceTrainer.Train('--input=combinedbig.txt --model_prefix=combbig --vocab_size=16000 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=[<PAD>] --unk_piece=[<UNK>] --bos_piece=[<SOS>] --eos_piece=[<EOS>] --normalization_rule_name=nfkc_cf')
# [7] END

# [8] START
sp = spm.SentencePieceProcessor()
# [8] END

# [9] START
sami_tokenized, swedish_tokenized = tokenizer_sp(sami_model, sami_sent, swedish_model, swedish_sent)
# [9] END

# [10] START
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
# [10] END

# [11] START
train_tgts = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in train_indices if i in range(len(sami_tokenized))])
train_src = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in train_indices if i in range(len(sami_tokenized))])
val_tgts = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in val_indices if i in range(len(sami_tokenized))])
val_src = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in val_indices if i in range(len(sami_tokenized))])
test_tgts = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in test_indices if i in range(len(sami_tokenized))])
test_src = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in test_indices if i in range(len(sami_tokenized))])
train_dataset = TensorDataset(train_src, train_tgts)
val_dataset = TensorDataset(val_src, val_tgts)
test_dataset = TensorDataset(test_src, test_tgts)
# [11] END

# [12] START
# Training hyperparameters
num_epochs = 500
learning_rate = 3e-4
batch_size = 64
# [12] END

# [13] START
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)
# [13] END

# [14] START
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
sp.Load(swedish)
src_pad_idx = sp.pad_id()
# [14] END

# [15] START
model = Trans_model(
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
# [15] END

# [16] START
sp.Load(swedish_model)
criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
# [16] END

# [17] START
load_model = False
# [17] END

# [18] START
if load_model == True:
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])           
    optimizer.load_state_dict(checkpoint['optimizer'])
model.to(device)
# [18] END

# [19] START
sentence = "Sametinget är samens valda organ i Norge."
# [19] END

# [20] START
sentence2 = "Det är viktigt att den inhemska dimensionen i nationella och internationella forum skyddas."
# [20] END

# [21] START
scores = []
e_losses = []
e_val_losses = []
e_ppl = []
e_val_ppl = []
# [21] END

# [22] START
threshold = 10
step = 5

for epoch in range(num_epochs):
    #print(f"[Epoch {epoch} / {num_epochs}]")

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, device, sami_model, swedish_model
    )
    translated_sentence2 = translate_sentence(
        model, sentence2, device, sami_model, swedish_model
    )

    print(f"Translated example sentences: \n {translated_sentence} \n {translated_sentence2}")
    losses = []
    val_losses = []
    ppl = []
    val_ppl = []


    for input_batch, target_batch in tqdm(train_loader):
        model.train()
        input_batch = input_batch.transpose(0, 1)
        target_batch = target_batch.transpose(0, 1)
        # Get input and targets and get to cuda
        inp_data = input_batch.to(device)
        target = target_batch.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        losses.append(loss.item())
        perplexity = torch.exp(loss)
        ppl.append(perplexity.item())
        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

    model.eval()
    mean_loss = sum(losses) / len(losses)
    e_losses.append(mean_loss) #keep track of training loss for plotting
    mean_perplex = sum(ppl) / len(ppl)
    e_ppl.append(mean_perplex) #keep track of training perplexity for plotting
    
    for input_batch, target_batch in tqdm(val_loader):
        input_batch = input_batch.transpose(0, 1)
        target_batch = target_batch.transpose(0, 1)
        # Get input and targets and get to cuda
        inp_data = input_batch.to(device)
        target = target_batch.to(device)


        # Forward prop
        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        val_losses.append(loss.item())
        perplexity = torch.exp(loss)
        val_ppl.append(perplexity.item())

    val_mean_loss = sum(val_losses) / len(val_losses)
    val_mean_perplex = sum(val_ppl) / len(val_ppl)
    if epoch == 0:
        best_loss = val_mean_loss
        e_since_impr = 0
        best_score = 0
        bleu_val = 0
    e_val_losses.append(val_mean_loss) #keep track of validation loss for plotting
    e_val_ppl.append(val_mean_perplex) #keep track of validation perplexity for plotting
     
    if (epoch+1) % step == 0: # evaluate after every step
        bleu_train, chrf_train, ter_train = get_scores(train_src[:200], train_tgts[:200], model, device, sami_model, swedish_model, "greedy")
        print("Bleu score in train set in epoch", epoch + 1, ":", bleu_train, "chrf:", chrf_train, "ter:", ter_train)
        bleu_val, chrf_val, ter_val = get_scores(val_src, val_tgts, model, device, sami_model, swedish_model, "greedy")
        print("Bleu score in val set in epoch", epoch + 1, ":", bleu_val, "chrf:", chrf_val, "ter:", ter_val)
        scores.append(bleu_val) #keep track of validation bleu scores
        if bleu_val.score >= best_score:
            # if current state is best performing save checkpoint
            best_score = bleu_val.score
            e_since_impr = 0 # reset epochs with no improvement
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
        else:
            scheduler.step() 
            e_since_impr += 1 # keep track of epochs*step without improvement
            print("Epochs since improvement:", e_since_impr, "new learning rate:", scheduler.get_lr()[0])
            if e_since_impr >= threshold: # if no improvement for x epochs -> early stopping
                print("Epochs since improvement:", e_since_impr, ". Early Stopping at epoch", epoch)
                
    
    
    print("Current best score:", best_score)
    print("Loss in epoch", epoch + 1 , ":", mean_loss, ", perplexity", mean_perplex, "_____ Validation loss:", val_mean_loss, ", perplexity", val_mean_perplex)
# [22] END

# [23] START
get_scores(test_src, test_tgts, model, device, sami_model, swedish_model, "greedy")
# [23] END

# [24] START

# [24] END

