import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, path, mode):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained('./pretrain/pretrain')
        if path is not None and path != "None":
            self.bert.load_state_dict(torch.load(path)["bert-base"])
            print("We load " + path + " to train!")
        self.max_length = max_length
        self.mode = mode
        self.max_length_name = 8
        self.tokenizer = BertTokenizer.from_pretrained('./pretrain/pretrain/bert-base-uncased-vocab.txt')

    def forward(self, inputs, cat=True):
        #inputs dict 4
        #['word']  [40,128]
        #['mask']  [40,128]
        #["pos1"]  [40]
        #["pos2"]  [40]
        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
        #[encoded_layers, pooled_output] 
        #(batch_size， sequence_length，hidden_size) [40, 128, 768]
        # (batch_size, hidden_size) [40,768]
        if cat:
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]#[40,768]  头实体
            t_state = outputs[0][tensor_range, inputs["pos2"]]#[40, 768] 尾实体
            state = torch.cat((h_state, t_state), -1)
            return state, outputs[0]
        else:
            return outputs[1], outputs[0]

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
  #['[CLS]',
  # 'one','of', 'the', 'first', 'rotary', 'hydraulic', 'motors', 'to', 'be', 'developed', 'was',
  # 'that', 'constructed', 'by', 'william', 'armstrong', 'for', 'his', '[unused0]', 'swing',
  # 'bridge', '[unused2]','over','the','[unused1]','river','tyne','[unused3]','.']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
    
        # pos
        pos1 = np.zeros(self.max_length, dtype=np.int32)
        pos2 = np.zeros(self.max_length, dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length
    
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1
        
        if pos1_in_index == 0:
            pos1_in_index = 1
        if pos2_in_index == 0:
            pos2_in_index = 1
        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
    
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask# 头、尾实体起始位置

    def tokenize_rel(self, raw_tokens):
        # ['crosses','obstacle (body of water, road, ...) which this bridge crosses over or this tunnel goes under']
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        #['[CLS]', 'crosses', '[SEP]']
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # ['[CLS]','crosses','[SEP]','obstacle','(','body','of','water',',','road',',','.','.','.',')','which','this','bridge','crosses','over','or','this','tunnel','goes','under']
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_name(self, name):
        # for FewRel 2.0
        # token -> index
        tokens = ['[CLS]']
        for token in name.split('_'):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length_name:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length_name]

        # mask
        mask = np.zeros(self.max_length_name, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask
