import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import Queue
import sacrebleu
import xml.etree.ElementTree as ET

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

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def get_scores(enc_sources, enc_target_sents, model, device, tokenizersrc, tokenizertrg, search="greedy", n=4):
    """
    takes a list of sentences and their translations in string form and returns score objects
    model is the trained transformer model
    tokenizer is the spm sentencpiece vocabulary in the form "name".model
    search is the decoding strategy, either greedy or beam search
    n is the beam width in beam search
    """
    model.eval()
    sp.load(tokenizertrg)
    targets = []
    outputs = []
    target_str = [sp.DecodeIds(sent.tolist()) for sent in enc_target_sents]
    output_str = []
    if search == "greedy":
        x = divide_chunks(enc_sources, 1000)
        output_str = []
        for sents in x:
            y = translate_enc_sentences(model, sents, device, tokenizertrg, max_length=150)
            output_str.extend(y)
        bleu = sacrebleu.corpus_bleu(output_str, [target_str])
        chrf = sacrebleu.corpus_chrf(output_str, [target_str])
        ter = sacrebleu.corpus_ter(output_str, [target_str])
        
        return bleu, chrf, ter
    elif search == "beam":
        prediction = beam_search(source, device, tokenizersrc, tokenizertrg, n)
    sp.Load(tokenizertrg)
    target = sp.DecodeIds(target.tolist())
    targets.append([target.split()])
    target_str.append(target)
    outputs.append(prediction.split())
    output_str.append(prediction)
    
    bleu = sacrebleu.corpus_bleu(output_str, [target_str])
    chrf = sacrebleu.corpus_chrf(output_str, [target_str])
    ter = sacrebleu.corpus_ter(output_str, [target_str])
    return bleu, chrf, ter


sp = spm.SentencePieceProcessor()


class Node(object):
    """
    helper object in beam search decoding
    hidden saves the partial translation so far
    previous node is the node of the last decoded token
    decoder_input is the index of the current token
    log_prob is the probability of the current sequence
    length is the amount of tokens in the current sequence
    """
    def __init__(self, hidden, previous_node, decoder_input, log_prob, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input.unsqueeze(1)
        self.log_prob = log_prob
        self.length = length

        
def beam_search(model, src, device, tokenizer_src, tokenizer_trg, beam_width=3, max_length=150):
    """
    takes a sentence and returns the most probable translation sequence as found by beam search
    """
    model.eval()
    sp.Load(tokenizer_src)
    
    byte_pairs = sp.EncodeAsIds(src) #encode source sentence
    byte_pairs.insert(0, sp.bos_id()) #add bos token
    byte_pairs.append(sp.eos_id()) #add eos token
    byte_pairs.extend([sp.pad_id()] * (max_len - len(byte_pairs))) #padding
    input_tensor = torch.LongTensor(byte_pairs).unsqueeze(1).to(device)
    
    sp.Load(tokenizer_trg)
    in_text = sp.bos_id() #first token of output sequence
    trg_tensor = torch.LongTensor([sp.bos_id()]).to(device)
    
    node = Node(None, None, trg_tensor, 0, 1) #initialize node to keep track of sequences
    q = Queue()
    q.put(node)
    
    end_nodes = []
    while not q.empty():
        candidates = []
        # level traversal
        for _ in range(q.qsize()):
            node = q.get()
            decoder_input = node.decoder_input
            
            if node.hidden != None:
                decoder_input = torch.cat((node.hidden, decoder_input), 0) #decoder input = source and partial translation
            
            if decoder_input[-1] == sp.eos_id() or node.length >= max_length: #end if eos token is predicted or max len is reached 
                end_nodes.append(node)
                continue
                    
            output = model(input_tensor, decoder_input) #predict next token
            output = output.squeeze(1)
            
            log_prob = F.log_softmax(output, dim=1)
            
            log_prob, indices = log_prob[-1, :].topk(beam_width) #get top n next tokens and probabilities
            
            for k in range(beam_width):
                index = indices[k].unsqueeze(0)
                log_p = log_prob[k].item()
                child = Node(decoder_input, node, index, node.log_prob + log_p, node.length + 1)
                candidates.append((node.log_prob + log_p, child))
        
        candidates = sorted(candidates, key=lambda x:x[0], reverse=True) #sort candidates after sequence probability
        length = min(len(candidates), beam_width)
        for i in range(length):
            q.put(candidates[i][1]) #append top n probable sequences 
    
    candidates = []
    for node in end_nodes:
        value = node.log_prob
        candidates.append((value, node))
    
    candidates = sorted(candidates, key=lambda x:x[0], reverse=True)
    node = candidates[0][1]
    
    res = []
    while node.previous_node != None:
        res.append(sp.DecodeIds([node.decoder_input.item()]))
        node = node.previous_node

    return sp.DecodeIds(res[::-1])

        
def translate_sentence(model, sentence, device, tokenizer_src, tokenizer_trg, max_length=150):
    """
    takes a sentence and returns the most probable translation as found by greedy search
    """
    # Go through each token and convert to an index
    sp.Load(tokenizer_src)
    byte_pairs = sp.EncodeAsIds(sentence)
    byte_pairs.insert(0, sp.bos_id())
    byte_pairs.append(sp.eos_id())
    byte_pairs.extend([sp.pad_id()] * (max_length - len(byte_pairs)))

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(byte_pairs).unsqueeze(1).to(device)
    sp.load(tokenizer_trg)
    # Insert bos token
    outputs = [sp.bos_id()]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)
            output = output.squeeze(1)
        output = F.log_softmax(output, dim=1)
        topv, topi = output[-1, :].data.topk(1)
        best_guess = topi.tolist()[0]
        outputs.append(best_guess)
        if best_guess == sp.eos_id():
            break
    return sp.DecodeIds(outputs)


def translate_enc_sentences(model, sentences, device, tokenizertrg, max_length=150):
    final_sents = {}
     # Convert to Tensor
    sentence_tensor = sentences
    sentence_tensor = sentence_tensor.transpose(0,1)
    # Insert <SOS> token
    sp.Load(tokenizertrg)
    outputs = [[sp.bos_id()] for i in range(sentence_tensor.shape[1])]
    for i in range(max_length):
        trg_tensor = [torch.LongTensor(outputs[i]) for i in range(len(outputs))]
        trg_tensor = torch.stack(trg_tensor)
        trg_tensor = trg_tensor.transpose(0, 1)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)
        output.transpose(1,2)
        for e in range(len(outputs)):
            outpute = F.log_softmax(output[:, e])
            topv, topi = outpute[-1, :].data.topk(1)
            best_guess = topi.tolist()[0]
            outputs[e].append(best_guess)
            if best_guess == sp.eos_id():
                final_sents[e] = outputs[e]
            elif len(outputs[e]) == max_length:
                if e not in final_sents.keys():
                    final_sents[e] = outputs[e]
        if len(final_sents.keys()) == len(sentences):
            final_list = []
            for i in range(len(final_sents.keys())):
                if sp.eos_id() in final_sents[i]:
                    sent = final_sents[i][:(final_sents[i].index(sp.eos_id())+1)]
                final_list.append(sent)
            decoded = [sp.DecodeIds(sent) for sent in final_list]
            
            return  decoded

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def read_data(xml_file):
    swedish_sent = []
    sami_sent = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for i, child in enumerate(root):
        if i == 1:
            for i, elem in enumerate(child):
                for e, subelem in enumerate(elem):
                    lang = subelem.get('{http://www.w3.org/XML/1998/namespace}lang')
                    if lang == "sv":
                        for seg in subelem:
                            sentencesv = seg.text
                    elif lang == "sme":
                        for seg in subelem:
                            sentencesme = seg.text
                    elif lang == None:
                        lang = subelem.attrib['lang']
                        if lang == "sv":
                            for seg in subelem:
                                sentencesv = seg.text                                
                        elif lang == "sme":
                            for seg in subelem:
                                sentencesme = seg.text
                    if e%2 != 0:
                        if sentencesv != None:
                            if sentencesme != None:
                                sami_sent.append(sentencesme)
                                swedish_sent.append(sentencesv)
    return sami_sent, swedish_sent

