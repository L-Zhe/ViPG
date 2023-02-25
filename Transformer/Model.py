import  torch
from    torch import nn
from    torch.nn import functional as F
from    .Module import (
    Encoder, Decoder, Embedding,
    PositionWiseFeedForwardNetworks,
    MultiHeadAttention, EncoderCell,
    DecoderCell, Linear)
from    copy import deepcopy
from    .SearchStrategy import SearchMethod
from    utils.tools import move2cuda, triu_mask
from    numpy import inf


class MultiModalAttn(nn.Module):
    def __init__(self, d_model, d_len):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(d_len, d_model))
        self.w_out = Linear(2 * d_model, d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        
    def forward(self, input, mask):
        score = torch.matmul(self.Q, input.transpose(-1, -2))
        if mask is not None:
            score.masked_fill_(mask, -inf)
        weight = F.softmax(score, dim=-1)
        input = torch.matmul(weight, input)
        return self.norm_attn(F.relu(input))

class MultiModalSpace(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_mu = Linear(d_model, d_model)
        self.w_log_var = Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, matrix):
        mu = self.w_mu(matrix)
        log_var = self.w_log_var(matrix)
        batch_size = mu.size(0)
        z = torch.randn(batch_size, 1, self.d_model).to(matrix.device) * torch.exp(log_var / 2) + mu
        kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1)
        kl_div = kl_div.mean(dim=-1)
        return z, kl_div.mean()


def copy(prob, attn_weight, p_g, src_index):        
    src_index = src_index.unsqueeze(1).repeat(1, prob.size(1), 1)
    attn_weight = (1 - p_g) * attn_weight
    prob = prob * p_g
    return prob.scatter_add(2, src_index, attn_weight)


def cal_attn(input, environ, mask):
    score = torch.matmul(input, environ.transpose(-1, -2))
    weight = F.softmax(score.masked_fill(mask, -inf), dim=-1)
    return weight


class transformer(nn.Module):
    def __init__(self, config):
        d_model = config.embedding_dim
        num_head = config.num_head
        num_layer_encoder = config.num_layer_encoder
        num_layer_decoder = config.num_layer_decoder
        d_ff = config.d_ff
        dropout_embed = config.dropout_embed
        dropout_sublayer = config.dropout_sublayer
        share_embed = config.share_embed
        PAD_index = config.PAD_index
        UNK_index = config.UNK_index
        position_method = config.position_method
        max_src_position = config.max_src_position
        max_tgt_position = config.max_tgt_position
        super().__init__()
        assert d_model % num_head == 0, \
            ("Parameter Error, require embedding_dim % num head == 0.")

        d_qk = d_v = d_model // num_head
        attention = MultiHeadAttention(d_model, d_qk, d_v, num_head)
        FFN = PositionWiseFeedForwardNetworks(d_model, d_model, d_ff)

        if share_embed:
            vocab_size = config.vocab_size
            self.src_embed = Embedding(vocab_size,
                                       d_model,
                                       dropout=dropout_embed,
                                       padding_idx=PAD_index,
                                       position_method=position_method,
                                       max_length=max(max_src_position, max_tgt_position))
            self.tgt_embed = self.src_embed
            self.tgt_mask = triu_mask(max(max_src_position, max_tgt_position))

        else:
            src_vocab_size = config.src_vocab_size
            tgt_vocab_size = config.tgt_vocab_size
            self.src_embed = Embedding(src_vocab_size,
                                       d_model,
                                       dropout=dropout_embed,
                                       padding_idx=PAD_index,
                                       position_method=position_method,
                                       max_length=max_src_position)
            self.tgt_embed = Embedding(tgt_vocab_size,
                                       d_model,
                                       dropout=dropout_embed,
                                       padding_idx=PAD_index,
                                       position_method=position_method,
                                       max_length=max_tgt_position)
            self.tgt_mask = triu_mask(max_tgt_position)
            vocab_size = tgt_vocab_size
        normalize_before = config.normalize_before

        self.Encoder = Encoder(d_model=d_model, 
                               num_layer=num_layer_encoder,
                               layer=EncoderCell(d_model,
                                                 deepcopy(attention),
                                                 deepcopy(FFN),
                                                 dropout_sublayer,
                                                 normalize_before),
                               normalize_before=normalize_before)
                                
        self.Decoder = Decoder(d_model=d_model,
                               vocab_size=vocab_size,
                               layer=DecoderCell(d_model,
                                                 deepcopy(attention),
                                                 deepcopy(FFN),
                                                 dropout_sublayer,
                                                 normalize_before),
                               num_layer=num_layer_decoder,
                               normalize_before=normalize_before)
        self.w_pg = Linear(d_model, 1)
        self.project = Linear(d_model, vocab_size)
        try:
            beam = config.beam
            search_method = config.decode_method
            self.decode_search = SearchMethod(search_method=search_method,
                                              BOS_index=config.txt_BOS_index,
                                              EOS_index=config.EOS_index,
                                              UNK_index=UNK_index,
                                              beam=beam)
        except:
            None
        self.PAD_index = PAD_index
        self.UNK_index = UNK_index
        self.POS = nn.Parameter(torch.randn(1, 1, d_model))
        self.IMG = nn.Parameter(torch.randn(1, 1, d_model))
        self.RELATION = nn.Parameter(torch.randn(1, 1, d_model))

    def generate(self, tgt_embed, encoder_output, src_mask, memory=None, src_index=None, max_dict_length=None, object_mask=None):
        output, memory, attn_weight = self.Decoder.generate(tgt_embed,
                                                            encoder_output,
                                                            src_mask,
                                                            memory)
        p_g = torch.sigmoid(self.w_pg(output))
        attn_weight = cal_attn(output, encoder_output[:, :max_dict_length, :], object_mask)
        prob = copy(F.softmax(self.project(output), dim=-1), attn_weight, p_g, src_index)
        return prob[:, -1:, :], memory, attn_weight, src_index

    def forward(self, **kwargs):

        assert kwargs['mode'] in ['train', 'test']

        if kwargs['mode'] == 'train':
            img_feature = move2cuda(kwargs['img_input']) + self.IMG
            src_mask = move2cuda(kwargs['src_mask'])
            self_mask = move2cuda(kwargs['self_mask'])
            max_dict_length = kwargs['max_dict_length']
            source = move2cuda(kwargs['source'])
            pos_mask = move2cuda(kwargs['pos_mask'])
            rand_mask = move2cuda(kwargs['rand_mask'])
            batch_size, length = source.size(0), source.size(1)
            tag_embed = self.POS.repeat(batch_size, length, 1).masked_fill(pos_mask, 0) + self.RELATION.repeat(batch_size, length, 1).masked_fill(pos_mask == 0, 0)
            input = torch.cat((img_feature, self.src_embed(source) + tag_embed), dim=1)
            feature = self.Encoder(input, self_mask)
            # dict_feature = txt_feature[:, :max_dict_length, :]
            # img_dict_mask = torch.cat((move2cuda(torch.zeros(batch_size, 1, img_feature.size(1)).bool()), self_mask[:, :1, :max_dict_length]), dim=-1)
            # img_dict_feature = torch.cat((img_feature, dict_feature), dim=1)
            img_dict_feature = feature[:, :197 + max_dict_length, :]
            txt_dict_feature = feature[:, 197:, :]
            tgt_len = kwargs['target_txt'].size(-1)
            target_img = move2cuda(kwargs['target_img'])
            target_txt = move2cuda(kwargs['target_txt'])
            img_dict_mask = self_mask[:, :1, :197 + max_dict_length]
            output_vis, _ = self.Decoder(self.tgt_embed(target_img),
                                         img_dict_feature,
                                         img_dict_mask,
                                         self.tgt_mask[:, :tgt_len, :tgt_len].cuda())
            output_cap, _ = self.Decoder(self.tgt_embed(target_txt),
                                         txt_dict_feature,
                                         src_mask,
                                         self.tgt_mask[:, :tgt_len, :tgt_len].cuda())
            object_feature = feature[:, 197:(197 + max_dict_length), :]
            object_mask = self_mask[:, 197:198, 197:(197 + max_dict_length)]
            attn_vis = cal_attn(output_vis, object_feature, object_mask)
            attn_cap = cal_attn(output_cap, object_feature, object_mask)
            copy_index = source[:, :max_dict_length]
            mask_cap = (torch.randn(copy_index.size()).to(copy_index.device) < 0.2).masked_fill(rand_mask, False)
            mask_vis = (torch.randn(copy_index.size()).to(copy_index.device) < 0.2).masked_fill(rand_mask, False)
            copy_index_vis = copy_index.masked_fill(mask_vis, self.UNK_index)
            copy_index_cap = copy_index.masked_fill(mask_cap, self.UNK_index)
            p_g_vis = torch.sigmoid(self.w_pg(output_vis))
            prob_vis = copy(F.softmax(self.project(output_vis), dim=-1), attn_vis, p_g_vis, copy_index_vis)
            p_g_cap = torch.sigmoid(self.w_pg(output_cap))
            prob_cap = copy(F.softmax(self.project(output_cap), dim=-1), attn_cap, p_g_cap, copy_index_cap)
            # prob_cap = F.softmax(self.project(output_cap), dim=-1)
            # return self.project(output_vis), prob_cap
            return prob_vis, prob_cap
            # return self.project(output_vis), self.project(output_cap)

        else:
            src_mask = move2cuda(kwargs['src_mask'])
            self_mask = move2cuda(kwargs['self_mask'])
            max_dict_length = kwargs['max_dict_length']
            source = move2cuda(kwargs['source'])
            # noun_tag = move2cuda(kwargs['noun_tag'])
            # noun_mask = move2cuda(kwargs['noun_mask'])
            pos_mask = move2cuda(kwargs['pos_mask'])
            batch_size, length = source.size(0), source.size(1)
            tag_embed = self.POS.repeat(batch_size, length, 1).masked_fill(pos_mask, 0) + self.RELATION.repeat(batch_size, length, 1).masked_fill(pos_mask == 0, 0)
            txt_feature = self.Encoder(self.src_embed(source) + tag_embed, self_mask)
            max_length = kwargs['max_length']
            return self.decode_search(decoder=self.generate,
                                      tgt_embed=self.tgt_embed.single_embed,
                                      src_mask=src_mask,
                                      encoder_output=txt_feature,
                                      max_length=max_length,
                                      src_index=source[:, :max_dict_length],
                                      max_dict_length=max_dict_length,
                                      object_mask=self_mask[:, :1, :max_dict_length])
