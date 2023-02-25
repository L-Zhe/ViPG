from numpy.lib.function_base import copy
import  torch
from    math import inf
from    torch.nn import functional as F


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=0, min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class SearchMethod:

    def __init__(self, search_method, BOS_index, EOS_index, UNK_index = None, beam=5):
        assert search_method in ['greedy', 'beam']
        self.BOS_index = BOS_index
        self.EOS_index = EOS_index
        self.search_method = search_method
        self.beam = beam
        self.UNK_index = UNK_index
        self.return_prob = False

    def __call__(self, *args, **kwargs):

        if self.search_method == 'greedy':
            return self.greedy_search(*args, **kwargs)
        else:
            return self.beam_search(*args, **kwargs)

    @torch.no_grad()
    def greedy_search(self, decoder, tgt_embed, src_mask, max_dict_length,
                      encoder_output, max_length, object_mask, src_index=None):
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        sentence = torch.LongTensor([self.BOS_index] * batch_size).reshape(-1, 1).to(device)
        end_flag = torch.BoolTensor(batch_size).fill_(False).to(device)
        i = 0
        memory = None
        if self.return_prob:
            total_prob = torch.FloatTensor().cuda()
        total_copy_index = None
        while i < max_length:
            embed = tgt_embed(sentence[:, -1:], i)
            prob, memory, copy_prob, copy_word = decoder(embed,
                                                         encoder_output,
                                                         src_mask,
                                                         memory,
                                                         src_index,
                                                         max_dict_length,
                                                         object_mask)
            # prob = embed[:, -1:, :]
            # if self.return_prob:
            #     total_prob = torch.cat((total_prob, F.softmax(prob, dim=-1)), dim=1)
            prob = prob.squeeze(1)
            # if i == 0:
            # prob = top_k_top_p_filtering(prob, top_k=5, top_p=0.9)
            # word = prob.multinomial(1).long().view(-1, 1)
            # else:
            word = prob.max(dim=-1)[1].long().view(-1, 1)
            prob, word = prob.topk(1, dim=-1, largest=True)
            copy_prob, copy_index = copy_prob.topk(1, dim=-1, largest=True)
            if total_copy_index is None:
                total_copy_index = copy_index.squeeze(-1)
            else:
                total_copy_index = torch.cat((total_copy_index, copy_index.squeeze(-1)), dim=1)
            # UNK_mask = word == self.UNK_index
            # copy_word = copy_word.gather(dim=-1, index=copy_index.squeeze(-1)).masked_fill(UNK_mask == 0, 0)
            # word.masked_fill_(UNK_mask, 0)
            # word += copy_word
            # index = prob.multinomial(1)
            # index = index.repeat(1, 5)
            # word = word.gather(dim=-1, index=index)[:, :1]
            sentence = torch.cat((sentence, word), dim=1)
            mask = (word == self.EOS_index).view(-1).masked_fill_(end_flag, False)
            end_flag |= mask
            if (end_flag == False).sum() == 0:
                break
            i += 1
        del memory, embed, mask, end_flag
        if self.return_prob:
            return sentence, total_prob, total_copy_index
        else:
            return sentence, total_copy_index

    # def beam_search(self, decoder, tgt_embed, src_mask, 
    #                 encoder_output, max_length):

    #     batch_size = encoder_output.size(0)
    #     device = encoder_output.device
    #     srcLen = encoder_output.size(1)
    #     # generate first word.
    #     sentence = torch.LongTensor(batch_size, 1).fill_(self.BOS_index).to(device)
    #     embed = tgt_embed(sentence)
    #     seq_mask = triu_mask(1).to(device)
    #     embed = decoder(embed, 
    #                     encoder_output, 
    #                     src_mask, 
    #                     seq_mask)
    #     prob = F.log_softmax(embed[:, -1, :], dim=-1)
    #     bos_mask = torch.BoolTensor(1, prob.size(-1)).cuda().fill_(False)
    #     bos_mask[0, self.EOS_index] = True
    #     bos_mask = bos_mask.repeat(batch_size, 1)
    #     prob.masked_fill_(bos_mask, -inf)
    #     totalProb, word = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True)

    #     sentence = sentence.unsqueeze(1).repeat(1, self.beam, 1).view(batch_size * self.beam, -1)
    #     sentence = torch.cat((sentence, word.view(-1, 1)), dim=-1)
    #     eos_flag = (word == self.EOS_index)
    #     # generate other word
    #     encoder_output = encoder_output.unsqueeze(1).repeat(1, self.beam, 1, 1)
    #     encoder_output = encoder_output.view(batch_size * self.beam, srcLen, -1)
    #     src_mask = src_mask.unsqueeze(1).repeat(1, self.beam, 1, 1)
    #     src_mask = src_mask.view(batch_size * self.beam, 1, -1)

    #     i = 1
    #     while i < max_length:

    #         embed = tgt_embed(sentence)    # [Batch*beam, 1, hidden]

    #         seq_mask = triu_mask(i + 1).to(device)
    #         embed = decoder(embed, 
    #                         encoder_output, 
    #                         src_mask, 
    #                         seq_mask)
    #         prob = F.log_softmax(embed[:, -1, :], dim=-1)
    #         flatten_eos_index = eos_flag.view(batch_size * self.beam, 1)
    #         vocab_size = prob.size(-1)
    #         mask = flatten_eos_index.repeat(1, vocab_size)
    #         prob.masked_fill_(mask, -inf)

    #         for j in range(mask.size(0)):
    #             if flatten_eos_index[j, 0]:
    #                 mask[j, 1:] = False
    #         prob.masked_fill_(mask, 0)
    #         prob += totalProb.view(-1, 1)
    #         totalProb, index = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True, sorted=True)
    #         # prob, index: [batch_size, beam]
    #         word = index % vocab_size
    #         index = index // vocab_size
    #         # print(index)
    #         eos_flag = eos_flag.gather(dim=-1, index=index)
    #         eos_flag |= (word == self.EOS_index)
    #         if eos_flag.sum() == batch_size * self.beam or \
    #            eos_flag[:, 0].sum() == batch_size:
    #             break
    #         sentence = sentence.view(batch_size, self.beam, -1)
    #         sentence = sentence.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i + 1))
    #         sentence = torch.cat((sentence, word.unsqueeze(-1)), dim=-1)
    #         sentence = sentence.view(batch_size * self.beam, -1)
    #         i += 1
    #     index = totalProb.max(1)[1]
    #     sentence = sentence.view(batch_size, self.beam, -1)
    #     output = torch.LongTensor().to(device)
    #     for i in range(batch_size):
    #         sent = sentence[i:i+1, index[i], :]
    #         output = torch.cat((output, sent), dim=0)
    #     return output

    @torch.no_grad()
    def beam_search(self, decoder, tgt_embed, src_mask, max_dict_length,
                    encoder_output, max_length, object_mask, src_index=None):

        batch_size = encoder_output.size(0)
        device = encoder_output.device
        srcLen = encoder_output.size(1)
        # generate first word.
        sentence = torch.LongTensor(batch_size, 1).fill_(self.BOS_index).to(device)
        embed = tgt_embed(sentence, 0)
        # seq_mask = triu_mask(1).to(device)
        # embed, memory = decoder(embed, 
        #                         encoder_output, 
        #                         src_mask)
        memory = None
        prob, memory, copy_prob, copy_word = decoder(embed,
                                                     encoder_output,
                                                     src_mask,
                                                     memory,
                                                     src_index,
                                                     max_dict_length,
                                                     object_mask)
        prob = torch.log(prob.squeeze(1))
        vocab_size = prob.size(-1)
        bos_mask = torch.BoolTensor(1, prob.size(-1)).cuda().fill_(False)
        bos_mask[0, self.EOS_index] = True
        bos_mask = bos_mask.repeat(batch_size, 1)
        prob.masked_fill_(bos_mask, -inf)
        totalProb, word = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True)

        sentence = sentence.unsqueeze(1).repeat(1, self.beam, 1).view(batch_size * self.beam, -1)
        copy_prob, copy_index = copy_prob.topk(1, dim=-1, largest=True)
        copy_index = copy_index.repeat(1, self.beam, 1,).view(batch_size * self.beam, -1)
        sentence = torch.cat((sentence, word.view(-1, 1)), dim=-1)
        eos_flag = (word == self.EOS_index)
        # generate other word
        # memory: [batch, beam, num_layers, num_head, seq_length, dim, 2]
        num_layers, num_heads, _, dimension = memory.size()[1:5]
        memory = memory.unsqueeze(1).repeat(1, self.beam, 1, 1, 1, 1, 1)\
                       .view(batch_size * self.beam, num_layers, num_heads, 1, dimension, 2)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, self.beam, 1, 1)
        encoder_output = encoder_output.view(batch_size * self.beam, srcLen, -1)
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).repeat(1, self.beam, 1, 1)
            src_mask = src_mask.view(batch_size * self.beam, 1, -1)
        object_mask = object_mask.unsqueeze(1).repeat(1, self.beam, 1, 1)
        object_mask = object_mask.view(batch_size * self.beam, 1, -1)
        i = 1
        while i < max_length:
            embed = tgt_embed(sentence[:, -1:], i)   # [Batch*beam, 1, hidden]

            prob, memory, copy_prob, copy_word = decoder(embed,
                                                         encoder_output,
                                                         src_mask,
                                                         memory,
                                                         src_index,
                                                         max_dict_length,
                                                         object_mask)
            prob = torch.log(prob.squeeze(1))
            flatten_eos_index = eos_flag.view(batch_size * self.beam, 1)
            mask = flatten_eos_index.repeat(1, vocab_size)
            prob.masked_fill_(mask, -inf)
            
            for j in range(mask.size(0)):
                if flatten_eos_index[j, 0]:
                    mask[j, 1:] = False
            prob.masked_fill_(mask, 0)
            prob += totalProb.view(-1, 1)
            totalProb, index = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True, sorted=True)
            # prob, index: [batch_size, beam]
            word = index % vocab_size
            index = index // vocab_size
            eos_flag = eos_flag.gather(dim=-1, index=index)
            eos_flag |= (word == self.EOS_index)
            if eos_flag.sum() == batch_size * self.beam or \
               eos_flag[:, 0].sum() == batch_size:
                break
            sentence = sentence.view(batch_size, self.beam, -1)
            sentence = sentence.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i + 1))
            sentence = torch.cat((sentence, word.unsqueeze(-1)), dim=-1)
            sentence = sentence.view(batch_size * self.beam, -1)
            copy_prob = copy_prob.view(batch_size, self.beam, -1)
            copy_prob, c_index = copy_prob.topk(1, dim=-1, largest=True)
            # print(c_index.size())
            copy_index = copy_index.view(batch_size, self.beam, -1)
            copy_index = copy_index.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i))
            # print(copy_index.size())
            c_index = c_index.gather(dim=1, index=index.unsqueeze(-1))
            copy_index = torch.cat((copy_index, c_index), dim=-1)
            copy_index = copy_index.view(batch_size * self.beam, -1)
            # print(src_mask.size(), index.size())
            src_mask = src_mask.view(batch_size, self.beam, -1).gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, src_mask.size(-1)))
            src_mask = src_mask.view(batch_size * self.beam, 1, -1)
            object_mask = object_mask.view(batch_size, self.beam, object_mask.size(-2), -1).gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, object_mask.size(-2), object_mask.size(-1)))
            object_mask = object_mask.view(batch_size * self.beam, object_mask.size(-2), -1)
            memory = memory.view(batch_size, self.beam, num_layers, num_heads, i + 1, dimension, 2)
            index = index.view(batch_size, self.beam, 1, 1, 1, 1, 1)\
                         .repeat(1, 1, num_layers, num_heads, i + 1, dimension, 2)
            memory = memory.gather(dim=1, index=index)\
                           .view(batch_size * self.beam, num_layers, num_heads, i + 1, dimension, 2)
            # print(index.size())
            
            i += 1
        # index = totalProb.max(1)[1]
        sentence = sentence.view(batch_size, self.beam, -1)
        copy_index = copy_index.view(batch_size, self.beam, -1)
        # output = torch.LongTensor().to(device)
        # for i in range(batch_size):
        #     sent = sentence[i:i+1, 0, :]
        #     output = torch.cat((output, sent), dim=0)

        return sentence[:, 0, :], copy_index[:, 0, :]
