from    utils import constants
import  argparse
from    collections import Counter
from    math import inf
from    utils.tools import save_vocab


def create_vocab(args, file_list, vocab_num, min_freq, lower):
    def create_corpus(file):
        with open(file, 'r') as f:
            if lower:
                corpus = [line.strip('\n').lower() for line in f.readlines()]
            else:
                corpus = [line.strip('\n') for line in f.readlines()]
        return ' '.join(corpus).split()
    corpus = []
    for file in file_list:
        corpus.extend(create_corpus(file))
    
    word2index = {}
    index2word = {}
    word2index[constants.PAD_WORD] = constants.PAD_index
    index2word[constants.PAD_index] = constants.PAD_WORD
    word2index[constants.UNK_WORD] = constants.UNK_index
    index2word[constants.UNK_index] = constants.UNK_WORD
    word2index[constants.img_BOS_WORD] = constants.img_BOS_index
    index2word[constants.img_BOS_index] = constants.img_BOS_WORD
    word2index[constants.txt_BOS_WORD] = constants.txt_BOS_index
    index2word[constants.txt_BOS_index] = constants.txt_BOS_WORD
    word2index[constants.EOS_WORD] = constants.EOS_index
    index2word[constants.EOS_index] = constants.EOS_WORD
    word2index[args.pos_dict_word] = len(word2index)
    index2word[len(index2word)] = args.pos_dict_word
    word2index[args.relation_word] = len(word2index)
    index2word[len(index2word)] = args.relation_word
    if vocab_num != -1 and min_freq != 0:
        w_count = [pair[0] for pair in Counter(corpus).most_common(vocab_num) if pair[1] >= min_freq]
    elif vocab_num != -1:
        w_count = [pair[0] for pair in Counter(corpus).most_common(vocab_num)]
    elif min_freq != 0:
        w_count = [k for k, v in Counter(corpus).items() if v >= min_freq]
    else:
        w_count = set(corpus)
    for word in w_count:
        if word2index.get(word) is None:
            word2index[word] = len(word2index)
            index2word[len(index2word)] = word
    return word2index, index2word


def main_vocab():
    parser = argparse.ArgumentParser(description='Create vocabulary.',
                                     prog='creat_vocab')

    parser.add_argument('-f', '--file', type=str, nargs='+',
                        help='File list to generate vocabulary.')
    parser.add_argument('--vocab_num', type=int, nargs='?', default=-1,
                        help='Total number of word in vocabulary.')
    parser.add_argument('--min_freq', type=int, default=0)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--save_path', type=str, default='./',
                        help='Path to save vocab.')
    parser.add_argument('--relation_word', type=str, default='<relation>')
    parser.add_argument('--pos_dict_word', type=str, default='<pos_dict>')
                                                
    args = parser.parse_args()
    if args.lower:
        args.relation_word = args.relation_word.lower()
        args.pos_dict_word = args.pos_dict_word.lower()
    word2index, index2word = create_vocab(args,
                                          file_list=args.file,
                                          vocab_num=args.vocab_num,
                                          min_freq=args.min_freq,
                                          lower=args.lower)
    print('Vocabulary Number: %d' % len(word2index))
    save_vocab(word2index, index2word, args.lower, word2index[args.relation_word], word2index[args.pos_dict_word], args.save_path)


if __name__ == '__main__':
    main_vocab()