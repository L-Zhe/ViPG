from    stanfordcorenlp import StanfordCoreNLP
from    tqdm import tqdm
import  nltk
from    numba import jit

path = '/home/linzhe/tools/stanford-corenlp-full-2018-10-05'
nlp = StanfordCoreNLP(path, lang='en')


def WORD_TO_POS(pos):
    word2poss = []
    pos2words = []
    for elem in pos:
        word2pos = dict(elem)
        # print(word2pos)
        posCount = {}
        for key in word2pos:
            word_pos = word2pos[key]
            if word_pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                if posCount.get(word_pos) is None:
                    posCount[word_pos] = 1
                    word2pos[key] = word_pos + '@0'
                else:
                    word2pos[key] = word_pos + '@' + str(posCount[word_pos])
                    posCount[word_pos] += 1
            else:
                word2pos[key] = 'KEEP'
        word2poss.append(word2pos)
        pos2words.append(dict(zip(word2pos.values(), word2pos.keys())))
    return word2poss, pos2words


def replacepos(sent, word2pos):
    pos_FLAG = ['<POS_DICT>']
    pos_DICT = {}
    for i in range(len(sent)):
        if word2pos.get(sent[i]) is not None and \
            word2pos[sent[i]] != 'KEEP':
            s = sent[i]
            sent[i] = word2pos[sent[i]]
            if pos_DICT.get(sent[i]) is None:
                pos_DICT[sent[i]] = True
                pos_FLAG.append(sent[i])
                pos_FLAG.append(s)

    return ' '.join(pos_FLAG) + ' <RELATION> ' + ' '.join(sent)


def posPaser(file):
    with open(file, 'r') as f:
        src_corpus = [line for line in f.readlines()]

    pos = []
    for i in tqdm(range(len(src_corpus))):
        pos.append(nlp.pos_tag(src_corpus[i]))
    # pos = nltk.pos_tag(src_corpus)
    # print(pos)
    word2pos, pos2word = WORD_TO_POS(pos)

    for i in range(len(src_corpus)):
        src_corpus[i] = replacepos(src_corpus[i].split(), word2pos[i])

    return src_corpus


if __name__ == '__main__':
    file = '/home/linzhe/data/para-nmt-50m/para-nmt-8k.src'
    save_file = '/home/linzhe/data/para-nmt-50m/para-nmt-8k.src.pos'
    source = posPaser(file)
    with open(save_file, 'w') as f:
        for line in source:
            f.write(line)
            f.write('\n')
