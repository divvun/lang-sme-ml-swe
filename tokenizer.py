import sentencepiece as spm

sp = spm.SentencePieceProcessor()

def tokenizer_sp(sami_model, sami_sent, swedish_model, swedish_sent, max_len=150):
    sp.Load(sami_model)
    sami_tokens = []
    long_sent_ids = []
    for i, sentence in enumerate(sami_sent):
        byte_pairs = sp.EncodeAsIds(sentence)
        byte_pairs.insert(0, sp.bos_id())
        byte_pairs.append(sp.eos_id())
        byte_pairs.extend([sp.pad_id()] * (max_len - len(byte_pairs)))
        if len(byte_pairs) > max_len:
            long_sent_ids.append(i)
        sami_tokens.append(byte_pairs)
    sp.Load(swedish_model)
    swedish_tokens = []
    for i, sentence in enumerate(swedish_sent):
        byte_pairs = sp.EncodeAsIds(sentence)
        byte_pairs.insert(0, sp.bos_id())
        byte_pairs.append(sp.eos_id())
        byte_pairs.extend([sp.pad_id()] * (max_len - len(byte_pairs)))
        if len(byte_pairs) > max_len:
            long_sent_ids.append(i)
        swedish_tokens.append(byte_pairs)
    unvalid_ids = list(set(long_sent_ids))
        
    sami_tokenized = []
    swedish_tokenized = []
    for i, sent in enumerate(sami_tokens):
        if i not in unvalid_ids:
            sami_tokenized.append(sent)
            swedish_tokenized.append(swedish_tokens[i])
    return sami_tokenized, swedish_tokenized