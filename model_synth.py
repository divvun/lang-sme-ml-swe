import logging
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=logging.DEBUG)

logging.info("Beginning to import dependencies.")


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sentencepiece as spm
from utils import *
from collections import Counter
from transformer_model import Transformer as Trans_model
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import json
import numpy as np

logging.info("Imported all dependencies.")

device = torch.device("cuda:0")

sami_model = "combbig.model"
swedish_model = "combbig.model"

model_path = "uni_joint_2layer_gelu_synth.pth.tar"


max_len = 150

logging.info("Reading smeswebig.tmx.gz")
sami_sent, swedish_sent = read_data('corpora/smeswebig.tmx.gz')

len(sami_sent)

logging.info("Loading sentence files")
add_sme_sent = []
with open("bound_sme_sent.txt") as f:
    for line in f:
        add_sme_sent.append(line)
with open("synth_sme_sent.txt") as f:
    for line in f:
        add_sme_sent.append(line)



add_swe_sent = []
with open('synth_swe_sent.txt') as f:
    for line in f:
        add_swe_sent.append(line)
with open('mono_swe_sent.txt') as f:
    for line in f:
        add_swe_sent.append(line)

len(add_swe_sent) == len(add_sme_sent)


#spm.SentencePieceTrainer.Train('--input=combinedbig.txt --model_prefix=combbig --vocab_size=16000 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=[<PAD>] --unk_piece=[<UNK>] --bos_piece=[<SOS>] --eos_piece=[<EOS>] --normalization_rule_name=nfkc_cf')

sp = spm.SentencePieceProcessor()

logging.info("Processing sentences")

sami_sent.extend(add_sme_sent)

swedish_sent.extend(add_swe_sent)


sami_tokenized, swedish_tokenized = tokenizer_sp(sami_model, sami_sent, swedish_model, swedish_sent)

logging.info("Loading indices")
with open("./indices.json", "r", encoding="utf-8") as f:
    ind = json.load(f)
train_indices = ind["train_indices"]
val_indices = ind["val_indices"]
test_indices = ind["test_indices"]
#with open('val_indices_big.txt') as fp:
#    for line in fp:
#        val_indices.append(int(line))
#with open('test_indices_big.txt') as fp:
#    for line in fp:
#        test_indices.append(int(line))
#for i in range(len(sami_tokenized)):
#    if i not in val_indices:
#        if i not in test_indices:
#            train_indices.append(i)

train_indices.extend([i for i in range(train_indices[-1]+1, len(sami_tokenized))+1])
# print(len(sami_tokenized))
# print(train_indices[-1])

logging.info("Creating stacks and datasets")
train_src = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in train_indices if i in range(len(sami_tokenized))])
train_tgts = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in train_indices if i in range(len(sami_tokenized))])
val_src = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in val_indices if i in range(len(sami_tokenized))])
val_tgts = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in val_indices if i in range(len(sami_tokenized))])
test_src = torch.stack([torch.LongTensor(sami_tokenized[i]) for i in test_indices if i in range(len(sami_tokenized))])
test_tgts = torch.stack([torch.LongTensor(swedish_tokenized[i]) for i in test_indices if i in range(len(sami_tokenized))])
train_dataset = TensorDataset(train_src, train_tgts)
val_dataset = TensorDataset(val_src, val_tgts)
test_dataset = TensorDataset(test_src, test_tgts)

num_epochs = 200
learning_rate = 3e-4
batch_size = 64


train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)

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

logging.info("Loading trans model")
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

sp.Load(swedish_model)
criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

load_model = False

if load_model == True:
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])           
    optimizer.load_state_dict(checkpoint['optimizer'])
model.to(device)


sentence = "Sámediggi lea sámiid álbmotválljen orgána Norggas."

sentence2 = "Deaŧalaš lea gozihit álgoálbmotoli nationála ja riikkaidgaskasaš forain."


scores = []
e_losses = []
e_val_losses = []
e_ppl = []
e_val_ppl = []

threshold = 5
step = 5

logging.info("Beginning training")


for epoch in range(num_epochs):
    #eprint(f"[Epoch {epoch} / {num_epochs}]")

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, device, sami_model, swedish_model
    )
    translated_sentence2 = translate_sentence(
        model, sentence2, device, sami_model, swedish_model
    )

    logging.info(f"Translated example sentences: \n {translated_sentence} \n {translated_sentence2}")
    losses = []
    val_losses = []
    ppl = []
    val_ppl = []


    for input_batch, target_batch in train_loader:
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
    
    for input_batch, target_batch in val_loader:
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
        logging.info("Bleu score in train set in epoch", epoch + 1, ":", bleu_train, "chrf:", chrf_train, "ter:", ter_train)
        bleu_val, chrf_val, ter_val = get_scores(val_src, val_tgts, model, device, sami_model, swedish_model, "greedy")
        logging.info("Bleu score in val set in epoch", epoch + 1, ":", bleu_val, "chrf:", chrf_val, "ter:", ter_val)
        scores.append(bleu_val) #keep track of validation bleu scores
        if bleu_val.score >= best_score:
            # if current state is best performing save checkpoint
            best_score = bleu_val.score
            e_since_impr = 0 # reset epochs with no improvement
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, model_path)
        else:
            scheduler.step() 
            e_since_impr += 1 # keep track of epochs*step without improvement
            logging.info("Epochs since improvement:", e_since_impr, "new learning rate:", scheduler.get_lr()[0])
            if e_since_impr >= threshold: # if no improvement for x epochs -> early stopping
                logging.info("Epochs since improvement:", e_since_impr, ". Early Stopping at epoch", epoch)
                
    
    
    logging.info("Current best score:", best_score)
    logging.info("Loss in epoch", epoch + 1 , ":", mean_loss, ", perplexity", mean_perplex, "_____ Validation loss:", val_mean_loss, ", perplexity", val_mean_perplex)

logging.info("scores:", scores)

logging.info(get_scores(test_src, test_tgts, model, device, sami_model, swedish_model, "greedy"))


