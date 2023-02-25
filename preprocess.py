# from FasySeq.utils.trainer import get_model
import  argparse

from    utils import constants
from    utils.tools import load_vocab
import  os
from    math import inf
import  torch
from    torch import LongTensor
import  pickle
from    tqdm import tqdm
from    utils.tools import pad_mask
from    PIL import Image, ImageFile
import  numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_invalid_date(data, args):
    max_src_position = args.max_src_position
    max_tgt_position = args.max_tgt_position
    discard_invalid_data = getattr(args, 'discard_invalid_data', False)
    source = data['source']
    target = data.get('target')
    if max_src_position != inf or max_tgt_position != inf:
        total_len = len(source)
        if discard_invalid_data:
            del_index = []
            for i in range(total_len):
                if len(source[i]) > max_src_position or \
                   (target and len(target[i]) > max_tgt_position - 1):
                    del_index.insert(0, i)
            if args.rank == 0:
                print("===> Discard invalid data: %d" % len(del_index))
            for index in del_index:
                del source[index]
                if target:
                    del target[index]
        else:
            for i in range(total_len):
                source[i] = source[i][:max_src_position]
                if target:
                    target[i] = target[i][:max_tgt_position - 1]

def replace_rel(seq, relation_index, idx2word):
    try:
        rel_index = seq.index(relation_index)
        i = 1
        pos_dict = {}
        while i < rel_index:
            j = i + 1
            pos_dict[seq[i]] = []
            while '@@' in idx2word[seq[j]] and j < rel_index:
                pos_dict[seq[i]].append(seq[j])
                j += 1
            if j == rel_index:
                break
            pos_dict[seq[i]].append(seq[j])
            i = j + 1
        # print(idx2word[seq[relation_index]])
        mask = []
        output = []
        for i in range(rel_index):
            if pos_dict.get(seq[i]) is not None:
                mask.append(True)
            else:
                mask.append(False)
        for i in range(rel_index + 1, len(seq)):
            if pos_dict.get(seq[i]) is None:
                output.append(seq[i])
            else:
                output.extend(pos_dict[seq[i]])
        return (output, mask)
    except:
        return seq

def extract_noun(seq, end_flag):
    noun_tag = []
    info_seq = []
    flag = False
    for i in range(len(seq)):
        flag |= seq[i] == end_flag
        if i == 0:
            info_seq.append(seq[i])
            noun_tag.append(seq[i])
        else:
            if (flag or i % 2 == 0):
                info_seq.append(seq[i])
            else:
                noun_tag.append(seq[i])
    return (info_seq, noun_tag)


def get_tokens(data, args):
    total_len = len(data)
    train_flag = args.mode == 'train'

    if train_flag:
        data.sort(key=lambda line: len(line[1]), reverse=True)
    else:
        data = [(index, value) for index, value in sorted(list(enumerate(data)), key=lambda line: len(line), reverse=True)]
  
    index_pair = []
    st = 0
    total_len = len(data)
    while st < total_len:
        if train_flag:
            max_length = len(data[st][1])
        else:
            max_length = len(data[st])
        ed = min(st + args.max_tokens // max_length, total_len)
        if ed == st:
            ed += 1
        index_pair.append((st, ed))
        st = ed
    process_data = []
    if train_flag:
        for (st, ed) in tqdm(index_pair):
            img_file, caption = zip(*data[st:ed])
            img_file = [os.path.join(args.img_path, file) for file in img_file]

            # source, noun_tag = zip(*[extract_noun(seq, args.relation_index) for seq in caption])
            relation_index = [seq.index(args.relation_index) for seq in caption]
            source = LongTensor(pad_batch(caption, args.PAD_index))
            max_len = max([len(seq) for seq in caption])
            # noun_tag = LongTensor([seq + [args.PAD_index] * (max_len - len(seq)) for seq in noun_tag])
            pos_mask = torch.zeros(source.size(0), source.size(1), 1).bool()
            for i in range(len(relation_index)):
                pos_mask[i, relation_index[i]:, :] = True
            src_mask = pad_mask(source, args.PAD_index)
            # noun_mask = (noun_tag == args.PAD_index).unsqueeze(-1)
            self_mask = torch.cat((torch.zeros(src_mask.size(0), src_mask.size(-1), 197).bool(), src_mask.repeat(1, src_mask.size(-1), 1)), dim=-1)
            self_mask = torch.cat((torch.zeros(src_mask.size(0), 197, self_mask.size(-1)).bool(), self_mask), dim=1)
            for i in range(len(relation_index)):
                self_mask[i, relation_index[i] + 197:, :197] = True
                self_mask[i, :197 + relation_index[i], relation_index[i] + 197:] = True
                self_mask[i, 197:197 + relation_index[i], :197] = True
            # tgt = [seq[index:] + seq[:index] for seq, index in list(zip(caption, relation_index))]
            tgt = [replace_rel(seq, args.relation_index, args.idx2word) for seq in caption]
            tgt, rand_mask = zip(*tgt)
            max_mask_length = max([len(m) for m in rand_mask])
            rand_mask = LongTensor([m + [True] * (max_mask_length - len(m)) for m in rand_mask]).bool()
            # print(source[0])
            # print(tgt[0])
            # print('\n')
            # print('*' * 20)
            tgt_input_img = [[args.img_BOS_index] + seq for seq in tgt]
            tgt_input_txt = [[args.txt_BOS_index] + seq for seq in tgt]
            tgt_output = [seq + [args.EOS_index] for seq in tgt]
            tgt_input_img = LongTensor(pad_batch(tgt_input_img, args.PAD_index))
            tgt_input_txt = LongTensor(pad_batch(tgt_input_txt, args.PAD_index))
            tgt_output = LongTensor(pad_batch(tgt_output, args.PAD_index))

            process_data.append((img_file, source, tgt_input_img, tgt_input_txt, tgt_output, pos_mask, self_mask, src_mask, rand_mask, max(relation_index)))
    else:
        for (st, ed) in tqdm(index_pair):
            rank, caption = zip(*data[st:ed])
            # source, noun_tag = zip(*[extract_noun(seq, args.relation_index) for seq in caption])
            relation_index = [seq.index(args.relation_index) for seq in caption]
            source = LongTensor(pad_batch(caption, args.PAD_index))
            # max_len = max([len(seq) for seq in source])
            # noun_tag = LongTensor([seq + [args.PAD_index] * (max_len - len(seq)) for seq in noun_tag])
            pos_mask = torch.zeros(source.size(0), source.size(1), 1).bool()
            for i in range(len(relation_index)):
                pos_mask[i, relation_index[i]:, :] = True
            src_mask = pad_mask(source, args.PAD_index)
            # noun_mask = (noun_tag == args.PAD_index).unsqueeze(-1)
            self_mask = src_mask.repeat(1, src_mask.size(-1), 1)
            for i in range(len(relation_index)):
                self_mask[i, :relation_index[i], relation_index[i]:] = True
            process_data.append((rank, source, pos_mask, self_mask, src_mask, max(relation_index)))
    return process_data


def restore_rank(data):
    data.sort(key=lambda x: x[0])
    return list(zip(*data))[1]


def get_data(args, data=None):
    if data is None:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
    max_src_len = data['max_src_len']
    max_tgt_len = max_src_len
    args.max_src_position = min(max_src_len, args.max_src_position)
    if max_tgt_len:
        args.max_tgt_position = max_tgt_len + 1

    # process_invalid_date(data=data,
    #                      args=args)
    data = get_tokens(data['caption_data'], args)
    return data, 1


def pad_batch(batch, pad_index):
    max_len = max(len(seq) for seq in batch)

    return [list(seq) + [pad_index] * (max_len - len(seq)) for seq in batch]


def save_data_loader(dataloader, save_file):

    save_path = os.path.join(*os.path.split(save_file)[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_file, 'wb') as f:
        pickle.dump(dataloader, f)


# class train_collate_fn:
#     def __init__(self, model_file):
#         self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_file)

#     @torch.no_grad()
#     def __call__(self, batch):
#         img_file, source, tgt_input_img, tgt_input_txt, tgt_output, pos_mask, self_mask, src_mask, max_dict_length = batch[0]
#         image = [Image.open(file).convert("RGB") for file in img_file]
#         img_input = self.feature_extractor(images=image, return_tensors='pt')
#         return {'mode': 'train',
#                 'img_input': img_input,
#                 'source': source,
#                 'target_img': tgt_input_img,
#                 'target_txt': tgt_input_txt,
#                 'src_mask': src_mask,
#                 'self_mask': self_mask,
#                 'pos_mask': pos_mask,
#                 'max_dict_length': max_dict_length}, \
#             tgt_output   

# from    nvidia.dali import pipeline_def, fn

def train_collate_fn(batch):
    img_file, source, tgt_input_img, tgt_input_txt, tgt_output, pos_mask, self_mask, src_mask, rand_mask, max_dict_length = batch[0]
    img_input = np.concatenate(tuple(np.load(file + '.feature.npy') for file in img_file), axis=0)
    return {'mode': 'train',
            'img_input': torch.from_numpy(img_input),
            'source': source,
            'target_img': tgt_input_img,
            'target_txt': tgt_input_txt,
            'src_mask': src_mask,
            'self_mask': self_mask,
            'pos_mask': pos_mask,
            'rand_mask': rand_mask,
            'max_dict_length': max_dict_length}, \
        tgt_output




def test_collate_fn(batch):
    rank, source, pos_mask, self_mask, src_mask, max_dict_length = batch
    return rank, {'source': source, 'pos_mask': pos_mask, 'self_mask': self_mask, 'src_mask': src_mask, 'max_dict_length': max_dict_length}


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--sent_file', type=str)
    parser.add_argument('--img_file', type=str, default=None)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--save_file', type=str)
    return parser.parse_args()


def data_process(filelist, word2index, lower):
    '''
    Change word to index. 
    '''
    data = []
    for file in filelist:
        with open(file, 'r', encoding='utf-8') as f:
            if lower:
                data.extend([line.strip('\n').lower().split() for line in f.readlines()])
            else:
                data.extend([line.strip('\n').split() for line in f.readlines()]) 

    def prepare_sequence(seq):
        return list(map(lambda word: word2index[constants.UNK_WORD] 
                        if word2index.get(word) is None else word2index[word], seq))
    return [prepare_sequence(seq) for seq in tqdm(data)]


def preprocess():
    args = get_args()
    word2index, _, lower, relation_index, pos_index = load_vocab(args.vocab)
    setattr(args, 'relation_index', relation_index)
    setattr(args, 'pos_index', pos_index)
    caption = data_process(filelist=[args.sent_file],
                           word2index=word2index,
                           lower=lower)
    max_src_len = max(len(seq) for seq in caption)
    if args.img_file:
        with open(args.img_file, 'r') as f:
            img = [file.strip('\n') for file in f.readlines()]
        caption = list(zip(img, caption))
    data = {'caption_data': caption,
            'max_src_len': max_src_len}

    save_data_loader(data, args.save_file)


if __name__ == '__main__':

    preprocess()
