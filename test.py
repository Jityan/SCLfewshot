import argparse
import os
import torch
import datetime

from torch.utils.data import DataLoader
from datasets.samplers import CategoriesSampler
from datasets.miniimagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.fc100 import FC100
from resnet import resnet12
from util import str2bool, set_gpu, seed_torch, compute_confidence_interval, normalize
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def get_dataset(args):
    if args.dataset == 'mini':
        testset = MiniImageNet('test', args.size)
        n_cls = 64
        print("=> MiniImageNet...")
    elif args.dataset == 'tiered':
        testset = TieredImageNet('test', args.size)
        n_cls = 351
        print("=> TieredImageNet...")
    elif args.dataset == 'cifarfs':
        testset = CIFAR_FS('test', args.size)
        n_cls = 64
        print("=> CIFAR-FS...")
    elif args.dataset == 'fc100':
        testset = FC100('test', args.size)
        n_cls = 60
        print("=> FC100...")
    else:
        print("Invalid dataset...")
        exit()

    test_sampler = CategoriesSampler(testset.label, args.test_batch,
                                    args.way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=args.worker, pin_memory=True)
    return test_loader, n_cls


def main(args):
    loader, n_cls = get_dataset(args)

    if args.dataset in ['mini', 'tiered']:
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls).cuda()
    elif args.dataset in ['cifarfs', 'fc100']:
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls).cuda()
    else:
        print("Invalid dataset...")
        exit()

    # check resume point
    checkpoint_file = os.path.join(args.save_path, 'max-acc.pth.tar')
    if not os.path.isfile(checkpoint_file):
        print("=> Model not found...")
        exit()
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    print("=> Model loaded...")
    
    model.eval()

    acc_list = []

    with torch.no_grad():
        for _, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]

            p = model(data_shot, is_feat=args.is_feat)
            q = model(data_query, is_feat=args.is_feat)

            py = torch.arange(args.way).repeat(args.shot)
            py = py.type(torch.LongTensor)
            qy = torch.arange(args.way).repeat(args.query)
            qy = qy.type(torch.LongTensor)

            if args.norm:
                p = normalize(p)
                q = normalize(q)
            
            p = p.detach().cpu().numpy()
            q = q.detach().cpu().numpy()
            py = py.view(-1).numpy()
            qy = qy.view(-1).numpy()
            # LR
            clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
            clf.fit(p, py)
            query_ys_pred_logit = clf.predict(q)
            acc_list.append(metrics.accuracy_score(qy, query_ys_pred_logit)*100)

    a, b = compute_confidence_interval(acc_list)
    print("{}-way {}-shot accuracy with 95% interval : {:.2f}±{:.2f}".format(args.way, args.shot, a, b))

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='./save/exp1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=1)
    # dataset
    parser.add_argument('--dataset', default='mini', choices=['mini','tiered','cifarfs','fc100'])
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--worker', type=int, default=8)
    # few-shot
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--norm', type=str2bool, nargs='?', default=True)
    parser.add_argument('--is-feat', type=str2bool, nargs='?', default=True)
    args = parser.parse_args()
    
    if args.dataset in ['mini', 'tiered']:
        args.size = 84
    elif args.dataset in ['cifarfs','fc100']:
        args.size = 32
        args.worker = 0
    # fix random seed
    seed_torch(args.seed)
    set_gpu(args.gpu)

    main(args)

    end_time = datetime.datetime.now()
    print("End time :{} total ({})".format(end_time, end_time - start_time))
