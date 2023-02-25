import  torch
from    utils.lang import translate2word
from    utils import constants
from    tqdm import tqdm
from    torch.utils.data import DataLoader
from    utils.args import get_generate_config
from    utils.tools import load_vocab
from    utils.makeModel import make_model
from    utils.tools import save2file
from    utils.checkpoint import load_model
from    utils.eval import Eval
from    preprocess import data_process, get_data, restore_rank
import  os


def _batch(args, st, ed):
    try:
        length = (args.source[st:ed] != args.PAD_index).sum(dim=-1)
        max_src_len = length.max().item()
        max_length = min(max_src_len * args.max_alpha, args.max_length)
        max_length = min(max_src_len + args.max_add_token, max_length)
        output, copy_index = args.model(source=args.source[st:ed, :max_src_len],
                                        src_mask=args.src_mask[st:ed, :, :max_src_len],
                                        self_mask=args.self_mask[st:ed, :max_src_len, :max_src_len],
                                        pos_mask=args.pos_mask[st:ed, :max_src_len, :],
                                        max_dict_length=args.max_dict_length,
                                        mode='test',
                                        max_length=max_length)
        output = output.tolist()
        copy_index = copy_index.tolist()
        for i in range(len(output)):
            output[i] = output[i][1:]
            if args.EOS_index in output[i]:
                end_index = output[i].index(args.EOS_index)
                index = min(int(args.max_alpha * length[i].item()), end_index)
                index = min(index, length[i] + args.max_add_token)
                output[i] = output[i][:index]

    except RuntimeError:
        if ed - st <= 1:
            raise RuntimeError
        print('==>Reduce Batch Size')
        torch.cuda.empty_cache()
        output = []
        copy_index = []
        length = max(int((ed - st) / 4), 1)
        while st < ed:
            _ed = min(st + length, ed)
            out, index = _batch(args, st, _ed)
            output.extend(out)
            copy_index.extend(index)
            st = _ed
    return output, copy_index

@torch.no_grad()
def generate(args):
    outputs = []
    copy_index = []
    rank = []
    args.model.eval()
    print('===>Start Generate.')
    for r, source, pos_mask, self_mask, src_mask, max_dict_length in tqdm(args.data):
        if source.dim() == 3:
            source = source[0]
            src_mask = src_mask[0]
            self_mask = self_mask[0]
            pos_mask = pos_mask[0]
            max_dict_length = max_dict_length[0]
            # noun_tag = noun_tag[0]
            # noun_mask = noun_mask[0]
        setattr(args, 'source', source)
        setattr(args, 'src_mask', src_mask)
        setattr(args, 'pos_mask', pos_mask)
        setattr(args, 'self_mask', self_mask)
        setattr(args, 'max_dict_length', max_dict_length)
        # setattr(args, 'noun_tag', noun_tag)
        # setattr(args, 'noun_mask', noun_mask)
        del source, src_mask
        output, index = _batch(args, 0, args.source.size(0))
        outputs.extend(output)
        copy_index.extend(index)
        rank.extend(r)
    return list(zip(rank, list(zip(translate2word(outputs, args.tgt_index2word), copy_index))))


def get_dataloader(args):
    data = None
    setattr(args, 'mode', 'test')
    if args.raw_file:
        source = data_process(filelist=[args.raw_file],
                              word2index=args.src_word2index,
                              lower=args.lower)
        del args.src_word2index
        max_src_len = max(len(seq) for seq in source)

        data = {'caption_data': source,
                'max_src_len': max_src_len}

    dataset, batch_size = get_data(args=args,
                                   data=data)
    dataset = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True)
    setattr(args, 'data', dataset)


def get_vocab_info(args):
    if args.file:
        if args.share_embed:
            _, tgt_index2word, lower, relation_index, pos_index = load_vocab(args.vocab)
            assert len(tgt_index2word) == args.vocab_size
        else:
            _, tgt_index2word, lower, relation_index, pos_index = load_vocab(args.tgt_vocab)
            assert len(tgt_index2word) == args.tgt_vocab_size
    else:
        if args.share_embed:
            src_word2index, tgt_index2word, lower, relation_index, pos_index = load_vocab(args.vocab)
            assert (len(src_word2index) == args.vocab_size)

        else:
            src_word2index, _, lower, relation_index, pos_index = load_vocab(args.src_vocab)
            _, tgt_index2word, _, relation_index, pos_index = load_vocab(args.tgt_vocab)
        assert (len(src_word2index) == args.src_vocab_size)
        assert (len(tgt_index2word) == args.tgt_vocab_size)
        setattr(args, 'src_word2index', src_word2index)
    setattr(args, 'tgt_index2word', tgt_index2word)
    setattr(args, 'lower', lower)
    setattr(args, 'relation_index', relation_index)
    setattr(args, 'pos_index', pos_index)
    if args.position_method == 'Embedding':
        if args.share_embed:
            args.max_length = min(args.max_length, max(args.max_src_position, args.max_tgt_position))
        else:
            args.max_length = min(args.max_length, args.max_tgt_position)


def replace_pos(seq):
    try:
        pos_index = seq.index('<pos_dict>')
        index = pos_index + 1
        pos_dict = {}
        while index < len(seq) - 1:
            pos_dict[seq[index]] = seq[index + 1]
            index += 2

        for i in range(1, index):
            if pos_dict.get(seq[i]) is not None:
                seq[i] = pos_dict[seq[i]]
        return ' '.join(seq[1:pos_index])
    except:
        return ' '.join(seq[1:])


def _main():

    args = get_generate_config()
    setattr(args, 'PAD_index', constants.PAD_index)
    setattr(args, 'UNK_index', constants.UNK_index)
    setattr(args, 'txt_BOS_index', constants.txt_BOS_index)
    setattr(args, 'img_BOS_index', constants.img_BOS_index)
    setattr(args, 'EOS_index', constants.EOS_index)
    setattr(args, 'rank', 0)
    assert (args.file is None) ^ (args.raw_file is None)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_num)
    model_state_dict, model_config = load_model(args.model_path)
    for key, value in model_config.items():
        setattr(args, key, value)
    print(args)
    get_vocab_info(args)
    model = make_model(args, model_state_dict, 0, False)
    setattr(args, 'model', model)
    get_dataloader(args)
    output = generate(args)
    output = restore_rank(output)
    output, copy_index = zip(*output)
    with open(args.raw_file, 'r') as f:
        corpus = [line.strip('\n').split() for line in f.readlines()]
    for i in range(len(output)):
        for j in range(len(output[i])):
            if output[i][j] == constants.UNK_WORD:
                output[i][j] = corpus[i][copy_index[i][j]]
    # output = [replace_pos(seq.split()) for seq in output]
    output = [' '.join(line).replace('@@', ' ').replace('  ', '') for line in output]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    save_file = os.path.join(args.output_path, 'result.txt')
    save2file(output, save_file)

    if args.ref_file is not None:
        eval = Eval(reference_file=args.ref_file)
        eval(save_file)


if __name__ == '__main__':
    _main()
