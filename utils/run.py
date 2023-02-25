import  torch
from    torch.nn import functional as F
import  time
import  functools
from    .tools import sync_between_gpu, move2cuda
from    copy import deepcopy
from    math import ceil
from    torch.utils.data import DataLoader
from    tqdm import tqdm


def print_info(func):
    @functools.wraps(func)
    def wrapper(args, model, info, input, target):
        total_tok = info['total_tok']
        cnt = info['cnt']
        total_loss = info['total_loss']
        ntoken = (input['source'] != args.PAD_index).sum().item()
        batch_size = input['source'].size(0)
        total_tok = ntoken
        cnt += batch_size
        loss = func(args,
                    input=input,
                    target=target,
                    model=model,
                    criterion=args.criterion)
        if total_loss is None:
            total_loss = deepcopy(loss)
            for key in total_loss.keys():
                total_loss[key] *= batch_size
        else:
            for key in total_loss.keys():
                total_loss[key] += loss[key] * batch_size
        # if args.batch % args.batch_print_info == 0:
        #     if args.world_size > 1:
        #         for key in total_loss.keys():
        #             total_loss[key] = sync_between_gpu(total_loss[key])
        #         total_tok = sync_between_gpu(total_tok)
        #         cnt = sync_between_gpu(cnt)
        #     total_time = time.time() - st_time
        #     st_time = time.time()
        #     if args.rank == 0:
        #         print(f'Batch: {args.batch}\t', end='')
        #         for key in total_loss.keys():
        #             print(key, f': {round(total_loss[key] / cnt, 5)}\t', end='')
        #         print(f'Tok pre Sec: {int(total_tok / total_time)}\t\tTime: {int(total_time)}')
        #     total_loss = None
        #     cnt = 0
        #     total_tok = 0
        # return {'total_loss': total_loss,
        #         'cnt': cnt,
        #         'total_tok': total_tok,
        #         'st_time': st_time}
        if args.world_size > 1:
            for key in total_loss.keys():
                total_loss[key] = sync_between_gpu(total_loss[key])
            total_tok = sync_between_gpu(total_tok)
            cnt = sync_between_gpu(cnt)
        return {'total_loss': total_loss,
                'cnt': cnt,
                'total_tok': total_tok
                }

    return wrapper


def mask_kl_div(q, p, mask):
    kl = (p * ((p / q).log())).sum(dim=-1)
    kl = kl.masked_fill(mask, 0)
    kl = kl.sum(dim=-1) / (mask == 0).sum(dim=1)
    return kl.mean()



@print_info
def step(args, input, target, model, criterion):
    prob_vis, prob_cap = model(**input)
    loss_ce_img = criterion(prob_vis, move2cuda(target))
    mask = move2cuda(target == args.PAD_index)
    # p_cap = F.softmax(prob_cap, dim=-1)
    # p_vis = F.softmax(prob_vis, dim=-1)
    # p_cap_t = F.softmax(prob_cap, dim=-1)
    # p_vis_t = F.softmax(prob_vis, dim=-1)
    loss_kl = mask_kl_div(prob_cap, prob_vis, mask) + mask_kl_div(prob_vis, prob_cap, mask)
    mask = move2cuda(target == args.PAD_index)
    norm = (mask == 0).sum(dim=-1)
    norm.masked_fill_(norm == 0, 1)
    loss_kl = loss_kl.masked_fill(mask, 0).sum(dim=-1) / norm
    loss_kl = loss_kl.mean()
    loss = loss_ce_img + loss_kl
    step_loss = {'loss_ce_img': loss_ce_img.item(),
                 'loss_kl': loss_kl.item()}
    loss.backward()
    return step_loss


def run(args):
    train_data = args.train_data
    optimizer = args.optimizer
    model = args.model
    model.train()
    info = {'total_loss': None,
            'cnt': 0,
            'total_tok': 0}
    optimizer.zero_grad()
    if args.rank == 0:
        loop = tqdm(range(len(train_data)))
        train_iter = iter(train_data)
        for _ in loop:
            st_time = time.time()
            input, target = next(train_iter)
            info = step(args,
                        model=model,
                        info=info,
                        input=input,
                        target=target)
            print_info = {'token_pre_sec': info['total_tok'] // (time.time() - st_time)}
            for key in info['total_loss'].keys():
                print_info[key] = round(info['total_loss'][key] / info['cnt'], 5)
            loop.set_description(f'EPOCH [{args.e}]')
            loop.set_postfix(**print_info)
            
            optimizer.update_optim()
        del train_iter
    else:
        for i, (input, target) in enumerate(train_data):
            info = step(args,
                        model=model,
                        info=info,
                        input=input,
                        target=target)
            optimizer.update_optim()
    optimizer.step()
    optimizer.zero_grad()


def fit(args, epoch):
    torch.backends.cudnn.benchmark = True
    rank = args.rank
    EPOCH = args.epoch
    checkpoint = args.checkpoint
    while epoch < EPOCH:
        setattr(args, 'e', epoch + 1)
        run(args)
        epoch += 1
        args.train_data.sampler.set_epoch(epoch)
        args.seed += epoch
        if rank == 0 and checkpoint is not None:
            checkpoint.save_point(model=args.model,
                                  optim=args.optimizer,
                                  epoch=epoch)
