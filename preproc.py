import json
import sentencepiece as spm
from utils import read_data, tokenizer_sp


sami_model = "combbig.model"
swedish_model = "combbig.model"

print("Reading smeswebig.tmx.gz")
sami_sent, swedish_sent = read_data('corpora/smeswebig.tmx.gz')

print("Loading sentence files")
add_sme_sent = []
with open("bound_sme_sent.txt") as f:
    for line in f:
        add_sme_sent.append(line)

add_swe_sent = []
with open('synth_swe_sent.txt') as f:
    for line in f:
        add_swe_sent.append(line)

len(add_swe_sent) == len(add_sme_sent)
sp = spm.SentencePieceProcessor()

print("Processing sentences")
sami_sent.extend(add_sme_sent)
swedish_sent.extend(add_swe_sent)
sami_tokenized, swedish_tokenized = tokenizer_sp(sami_model, sami_sent, swedish_model, swedish_sent)

train_indices = []
val_indices = []
test_indices = []


print("val_indices")
with open('val_indices_big.txt') as fp:
    for line in fp:
        val_indices.append(int(line))

val_set = set(val_indices)

print("test_indices")
with open('test_indices_big.txt') as fp:
    for line in fp:
        test_indices.append(int(line))

test_set = set(test_indices)

print("train_indices")
for i in range(len(sami_tokenized)):
    if i not in val_set and i not in test_set:
        train_indices.append(i)

print("writing output")
with open("indices.json", "w") as f:
    json.dump({
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }, f)
