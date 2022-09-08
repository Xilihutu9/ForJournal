# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:22:07 2021

@author: Administrator
"""

import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from torch.nn import init
from torch.autograd import Variable

class HCRP(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len#128
        self.rel_glo_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.temp_proto = 1  # moco 0.07
        self.dropout=nn.Dropout(0.1)
        self.r=5       
        self.a = torch.from_numpy(np.diag(np.ones(max_len - 1, dtype=np.int32),1)).cuda()
        #对角线上移一位
        self.b = torch.from_numpy(np.diag(np.ones(max_len, dtype=np.int32),0)).cuda()
        #对角线
        self.c = torch.from_numpy(np.diag(np.ones(max_len - 1, dtype=np.int32),-1)).cuda()
        #对角线下移一位
        self.tri_matrix = torch.from_numpy(np.triu(np.ones([max_len,max_len], dtype=np.float32),0)).cuda()
          

        self.weight_word = nn.Parameter(torch.Tensor(self.hidden_size, 1))
       # init.uniform_(self.weight_word , -0.1, 0.1)
        self.linear_first = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(self.hidden_size,self.r)
        self.linear_second.bias.data.fill_(0)
        #???初始化
        #init.uniform_(self.linear_first.weight , -0.1, 0.1)
        #init.uniform_(self.linear_second.weight , -0.1, 0.1)        
    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
    def compute_constituent(self,input ,mask ):
        
        score = input.masked_fill(mask == 0, -1e9)#[10,128,128]
        
        neibor_attn = F.softmax(score, dim=-1) #[10,128,128]
        
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-9)
        
        t = torch.log(neibor_attn + 1e-9).masked_fill(self.a==0, 0).matmul(self.tri_matrix)
        #[10,128,128]
        g_attn = self.tri_matrix.matmul(t).exp().masked_fill((self.tri_matrix.int()-self.b)==0, 0)     #只保留上三角
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(self.b==0, 1e-9)
        
        return g_attn
    
    def forward(self, support, query, rel_text, N, K, total_Q, is_eval=False):
        """
        :param support: Inputs of the support set. (B*N*K)
        :param query: Inputs of the query set. (B*total_Q)
        :param rel_text: Inputs of the relation description.  (B*N)
        :param N: Num of classes
        :param K: Num of instances for each class in the support set
        :param total_Q: Num of instances in the query set
        :param is_eval:
        :return: logits, pred, logits_proto, labels_proto, sim_scalar
        """
        support_glo, support_loc = self.sentence_encoder(support)  # (B * N * K, 2D), (B * N * K, L, D)
        #[40, 1536](头尾实体连接)   [40, 128, 768]（每个词向量）
        query_glo, query_loc = self.sentence_encoder(query)  # (B * total_Q, 2D), (B * total_Q, L, D)
        # #[40, 1536](头尾实体连接)   [40, 128, 768]（每个词向量）
        rel_text_glo, rel_text_loc = self.sentence_encoder(rel_text, cat=False)  # (B * N, D), (B * N, L, D)

        # global features
        ###################################################
        support_glo = support_glo.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        #[4, 10, 1, 1536]
        query_glo = query_glo.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)
        #[4,10,1536]
        rel_text_glo = self.rel_glo_linear(rel_text_glo.view(-1, N, self.hidden_size))  # (B, N, 2D)
        #[40,768]==>[4,10,1536]
        B = support_glo.shape[0] # 4
        # global prototypes  (需要,一个全局的特征)
        proto_glo = torch.mean(support_glo, 2) + rel_text_glo  # Calculate prototype for each class (B, N, 2D)
        #[4, 10, 1536]
        
        #========================================================
        penalty=0.0
     
        ################relation#############[10,128,768]
        #rel_text_loc=self.linear(rel_text_loc).contiguous()#(10,128,768)
        
        rel_support = torch.bmm(rel_text_loc, torch.transpose(support_loc, 2, 1))
        #[10,128,128]
        rel_general = torch.bmm(rel_text_loc, self.weight_word.unsqueeze(0).repeat(rel_text_loc.size(0),1,1))
        #[10,128,1]
        rel_word= rel_support + torch.transpose( rel_general,2,1)#[10,128,128]
      #  rel_score = torch.tanh(torch.div(rel_word, math.sqrt(self.hidden_size)))#[10,128,128] 
        
        
        '''rel word'''
        
        rel_score_word, _ = rel_word.max(-1) 
        rel_score_word = F.softmax(torch.tanh(rel_score_word), dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        #[ 40, 128, 1]                   
        rel_word_loc= (rel_score_word * rel_text_loc).contiguous()
        #[40, 128 ,768]
        rel_word_loc = torch.sum( rel_score_word * rel_text_loc, dim=1).view(B, N, K, self.hidden_size)  # (B * N * K, D)
        #[40,768]
        rel_word_loc = torch.mean(rel_word_loc, 2)  # (B, N, D)
        #[4,10,768]
        
        
        '''rel phase''' 
        rel_rel = torch.bmm(rel_text_loc, torch.transpose(rel_text_loc, 2, 1))
        rel_score1 = torch.div(rel_rel, math.sqrt(self.hidden_size))#[10,128,128] 
        
        #[B,1,128]
        rel_mask=rel_text['mask'].unsqueeze(1)&(self.a+self.c)#[10,128,128]        
        final_rel_phase_attn, penalty1  =self.phase_attn(rel_score1, rel_mask, rel_text_loc,rel_text['mask'])
       
        #[10,5,128]
        final_rel_phase_attn=self.dropout(final_rel_phase_attn)
        rel_phase_embeddings = final_rel_phase_attn@rel_text_loc     #矩阵相乘  [10,5,768]        
        rel_phase_embeddings=torch.sum(rel_phase_embeddings ,dim=1) / self.r #[10,768]
        #???????????
        rel_phase_loc=rel_phase_embeddings.view(B, N, K, self.hidden_size)        
        rel_phase_loc = torch.mean(rel_phase_loc, 2)#[4,10,768]
        
               
        #########################support 进行处理  support_loc [10,128,768]
        '''support word'''
        #[10,128,768] 
        support_rel = torch.bmm(support_loc, torch.transpose(rel_text_loc, 2, 1))
        #[10,128,128]
        support_general = torch.bmm(support_loc, self.weight_word.unsqueeze(0).repeat(support_loc.size(0),1,1))
        #[10,128,1]
        support_word= support_rel + torch.transpose( support_general,2,1)#[10,128,128]
        support_word = torch.tanh(torch.div(support_word, math.sqrt(self.hidden_size)))#[10,128,128] 
 

        support_score_word, _ = support_word.max(-1)          
        support_score_word = F.softmax(support_score_word, dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        support_score_word=self.dropout(support_score_word)#[40,128,1]
        sent_attn_vectors= (support_score_word* support_loc).contiguous()
        sent_attn_vectors = torch.sum(sent_attn_vectors, dim=1)  # (B * N * K, D)
        #[40,768]
        support_word_loc = sent_attn_vectors.view(B, N, K, self.hidden_size)#[4,10,1,768]
        
        '''support phase'''
        #自注意力
        support_support = torch.bmm(support_loc, torch.transpose(support_loc, 2, 1))#[10,128,128]
        support_score1 = torch.div(support_support, math.sqrt(self.hidden_size))#[10,128,128] 
        #[B,1,128]
        support_mask=support['mask'].unsqueeze(1)&(self.a+self.c)#[10,128,128]        
        
        final_support_phase_attn ,penalty2 =self.phase_attn(support_score1, support_mask, support_loc,support['mask'])
        #[10,5,128]
        final_support_phase_attn=self.dropout(final_support_phase_attn)        
        support_phase_embeddings = final_support_phase_attn@support_loc     #矩阵相乘  [10,5,768]        
        support_phase_embeddings=torch.sum(support_phase_embeddings ,dim=1)/ self.r   #[10,768]
        #???????????
        support_phase_loc=support_phase_embeddings.view(B, N, K, self.hidden_size)
        #[1,10,1,768]
        
        ###########query#####################
        
       # query_loc=self.linear(query_loc).contiguous()  
        query_query1 = torch.bmm(query_loc, torch.transpose(query_loc, 2, 1))  # (B * total_Q, L, L)
        #[40,128,128]
        query_general = torch.bmm(query_loc, self.weight_word.unsqueeze(0).repeat(query_loc.size(0),1,1))
        #[10,128,1]
        query_word= query_query1  + torch.transpose(query_general,2,1)#[10,128,128]

        query_word = torch.tanh(torch.div(query_word, math.sqrt(self.hidden_size)))#[10,128,128]         
     
        '''word attention'''
        query_score_word, _ = query_word .max(-1)          
        query_score_word = F.softmax(query_score_word, dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        query_score_word=self.dropout(query_score_word)#[40,128,1]
        query_word_vectors= (query_score_word* query_loc).contiguous()
        query_word_vectors = torch.sum(query_word_vectors, dim=1)  # (B * N * K, D)
        #[40,768]
        query_word_loc = query_word_vectors.view(B, total_Q, self.hidden_size)#[4,10,1,768]
    
        '''phase attention'''
        
        query_score1 = torch.div(query_query1, math.sqrt(self.hidden_size))#[10,128,128] 
        #[B,1,128]
        query_mask=query['mask'].unsqueeze(1)&(self.a+self.c)#[10,128,128]
        
        final_query_phase_attn,penalty3  =self.phase_attn(query_score1,query_mask, query_loc,query['mask'])
       #[10,5,128]
        final_query_phase_attn=self.dropout(final_query_phase_attn) 
        query_phase_embeddings = final_query_phase_attn@query_loc     #矩阵相乘  [10,5,768]        
        query_phase_embeddings=torch.sum(query_phase_embeddings ,dim=1) / self.r  #[10,768]
        query_phase_loc=query_phase_embeddings.view(B, total_Q, self.hidden_size) 
        #[1,10,768]
        
  
        penalty=(penalty1+penalty2+penalty3)/3
        
        ###傅里叶
        ''''''
        
        ########prototype

        
        proto_word_loc = torch.mean(support_word_loc, 2) + rel_word_loc  # (B, N, D)
        #[4,10,768]
        proto_phase_loc = torch.mean(support_phase_loc, 2) + rel_phase_loc
        
        # hybrid prototype
        proto_hyb = torch.cat((proto_glo, proto_word_loc,proto_phase_loc), dim=-1)  # (B, N, 3D)
        # [4,10,2304]
        query_hyb = torch.cat((query_glo, query_word_loc,query_phase_loc), dim=-1)  # (B, total_Q, 3D)
        #[4,10,2304]
        rel_text_hyb = torch.cat((rel_text_glo, rel_word_loc,rel_phase_loc), dim=-1)  # (B, N, 3D)
        #[4, 10, 2304]
        
        logits = self.__batch_dist__(proto_hyb, query_hyb)  # (B, total_Q, N)
        #[4,10,10]
        minn, _ = logits.min(-1) #[4,10]
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        #[4,10,11]
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        logits_proto, labels_proto, sim_scalar = None, None, None

        if not is_eval:
            # relation-prototype contrastive learning
            # # # relation as anchor
            rel_text_anchor = rel_text_hyb.view(B * N, -1).unsqueeze(1)  # (B * N, 1, 3D)
            #[40, 1, 2304]
            
            # select positive prototypes
            proto_hyb = proto_hyb.view(B * N, -1)  # (B * N, 3D)
            #[40,2304]
            
            pos_proto_hyb = proto_hyb.unsqueeze(1)  # (B * N, 1, 3D)
            #[40,1,2304]
            
            # select negative prototypes
            neg_index = torch.zeros(B, N, N - 1)  # (B, N, N - 1)
            #[4,10,9]
            
            for b in range(B):#4
                for i in range(N):#10
                    index_ori = [i for i in range(b * N, (b + 1) * N)]
                    index_ori.pop(i)
                    neg_index[b, i] = torch.tensor(index_ori)
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            neg_index = neg_index.long().view(-1).cuda()  # (B * N * (N - 1))
            #[360]
            neg_proto_hyb = torch.index_select(proto_hyb, dim=0, index=neg_index).view(B * N, N - 1, -1)
            #[40, 9 , 2304]
            # compute prototypical logits
            proto_selected = torch.cat((pos_proto_hyb, neg_proto_hyb), dim=1)  # (B * N, N, 3D)
            #[40,10,2304]
            
            logits_proto = self.__batch_dist__(proto_selected, rel_text_anchor).squeeze(1)  # (B * N, N)
            
            logits_proto /= self.temp_proto  # scaling temperatures for the selected prototypes

            # targets
            labels_proto = torch.cat((torch.ones(B * N, 1), torch.zeros(B * N, N - 1)), dim=-1).cuda()  # (B * N, 2N)
            
            # task similarity scalar
            features_sim = torch.cat((proto_hyb.view(B, N, -1), rel_text_hyb), dim=-1)
            #[4,10,4608]
            features_sim = self.l2norm(features_sim)
            sim_task = torch.bmm(features_sim, torch.transpose(features_sim, 2, 1))  # (B, N, N)
            #[4,10,10]
            sim_scalar = torch.norm(sim_task, dim=(1, 2))  # (B) [4]
            sim_scalar = torch.softmax(sim_scalar, dim=-1)
            sim_scalar = sim_scalar.repeat(total_Q, 1).t().reshape(-1)  # (B*totalQ)
            #[40]
        return logits, pred, logits_proto, labels_proto, sim_scalar , penalty

    def penalty(self,attention):
        penal=0.0
        #添加loss3
        attT = attention.transpose(1,2)#[10,128,3]
        identity = torch.eye(attention.size(1))#[3,3]
        identity = Variable(identity.unsqueeze(0).expand(attention.size(0),attention.size(1),attention.size(1))).cuda()
        #[512,10,10]
        penal = self.l2_matrix_norm(attention@attT - identity)/attention.size(0)   
        return penal
    
    #Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value
 
       
        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.DoubleTensor)
    
    
    def phase_attn(self,input_score, mask, input_loc,mask1):    
        
        #input_score 
        
        #连续性计算
        constituent=self.compute_constituent(input_score, mask) 
        #[10,128,128]
       
        #计算语义段
        seg = F.tanh(self.linear_first(input_loc))     #[10,128,768]  
        seg_attn= self.linear_second(seg)   #[10,128,5]    
        seg_score = F.softmax(seg_attn.transpose(1,2) ,dim=-1)   #[10,5,128]  ?????????
        penalty4 =self.penalty(seg_score)
        #??????mask
        mask1=mask1.unsqueeze(1).repeat(1,self.r,1)
        seg_score=seg_score.masked_fill(mask1==0, 0)
        seg_v, seg_i = seg_score.max(-1)#[10,5] [10,5]
        
        #1)计算通用虚词att  负相关
        general_phase_score = -torch.bmm(input_loc,self.weight_word.unsqueeze(0).repeat(input_loc.size(0),1,1)).squeeze(-1)#[10,128]
        #[10,128]
        general_phase_score_norm = F.softmax( general_phase_score,dim=-1).unsqueeze(1).repeat(1,self.r,1)
        #[10,5,128]
        
        #2)连续性
        cons_phase_score=torch.zeros(constituent.size(0),self.r,self.max_len)
        #[10,5,128]
        for i in range(constituent.size(0)):
            d= torch.index_select(constituent[i], dim = 0, index =seg_i[i])#[5,128]
            cons_phase_score[i]=d
       
        #3)语义相关性        
        related_phase_score=F.softmax(input_score,dim=-1)#[10,128,128]
        related_phase_score_norm=torch.zeros(constituent.size(0),self.r,self.max_len)
        #[10,5,128]
        for i in range(related_phase_score.size(0)):
            d= torch.index_select(related_phase_score[i], dim = 0, index =seg_i[i])#[5,128]
            related_phase_score_norm[i]=d
         
        
        final_phase_attn = seg_score +(cons_phase_score.cuda()+related_phase_score_norm.cuda()+general_phase_score_norm.cuda())/3#[10,768]
        
        return final_phase_attn , penalty4