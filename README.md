# lang-sme-ml-swe


Neural Machine Translation fro North Sámi to Swedish

data:

corpora/nobsme.tmx: Original Norwegian-North Sámi Corpus
corpora/smeswebig.tmx: Translated nobsme to Swedish

corpora/add_swe_sents.txt: Swedish Sentences from Anföranden
corpora/fof_sents.txt: Swedish Sentences from Forskning & Framsteg

mono_swe_sent.txt: Combined add_swe_sents and fof_sents
synth_sme_sent.txt: Translated mono_swe_sent to North Sámi with back-translation model

bound_sme_sent.txt: additional North Sámi sentences
synth_sme_sent.txt: Translated mono_sme_sent to Swedish with M1


.model and .vocab files: Unigram and BPE vocabularies


Models:

direction_layers_encoding_activation_parameter

e.g SMESWE-2layers-uni-gelu-synth.ipynb
from North Sámi to Swedish, 2 encoder and 2 decoder layers, unigram encoding, gelu activation, all synthetic data

Parts of the code were taken from this tutorial: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/seq2seq_transformer

Helper functions:

read_data.py: reads in tmx corpus file
utils.py: all helper functions