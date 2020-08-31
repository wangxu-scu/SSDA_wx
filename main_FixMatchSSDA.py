from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset, return_dataset_fixmatch
from utils.loss import entropy, adentropy
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=100000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='FIXMATCH',
                    choices=['S+T', 'ENT', 'MME', 'UODA', 'FIXMATCH'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--alpha', type=float, default=0.75, metavar='ALP',
                    help='value of alpha')
parser.add_argument('--beta', type=float, default=0.1, metavar='BETA',
                    help='value of beta')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')

parser.add_argument('--threshold', type=float, default=0.9, metavar='THRE',
                    help='value of threshold')


args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset_fixmatch(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num))

torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]


## F1: Source-based Classifier
## F1: Target-based Classifier
if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
    F2 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:

    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
    F2 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()
F2.cuda()

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu_weak = torch.FloatTensor(1)
im_data_tu_strong = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu_weak = im_data_tu_weak.cuda()
im_data_tu_strong = im_data_tu_strong.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu_weak = Variable(im_data_tu_weak)
im_data_tu_strong = Variable(im_data_tu_strong)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    F1.train()
    F2.train()
    optimizer_g = optim.SGD(params, lr=args.multi, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f1 = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f2 = optim.SGD(list(F2.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f1.zero_grad()
        optimizer_f2.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f1 = []
    for param_group in optimizer_f1.param_groups:
        param_lr_f1.append(param_group["lr"])
    param_lr_f2 = []
    for param_group in optimizer_f2.param_groups:
        param_lr_f2.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f1 = inv_lr_scheduler(param_lr_f1, optimizer_f1, step,
                                       init_lr=args.lr)
        optimizer_f2 = inv_lr_scheduler(param_lr_f2, optimizer_f2, step,
                                        init_lr=args.lr)
        lr = optimizer_f1.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu_weak.resize_(data_t_unl[0][0].size()).copy_(data_t_unl[0][0])
        im_data_tu_strong.resize_(data_t_unl[0][1].size()).copy_(data_t_unl[0][1])
        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t, im_data_tu_weak, im_data_tu_strong), 0)
        # target = torch.cat((gt_labels_s, gt_labels_t), 0)
        output = G(data)
        output_s = output[:len(im_data_s)]
        output_t = output[len(im_data_s):len(im_data_s)+len(im_data_t)]
        output_tu = output[len(im_data_s)+len(im_data_t):]
        output_tu_weak, output_tu_strong = output_tu.chunk(2)
        #out_1s = F1(output_s)
        out_2t = F2(output_t)

        out_2tu_weak = F2(output_tu_weak)
        out_2tu_strong = F2(output_tu_strong)
        pseudo_label_tu = torch.softmax(out_2tu_weak.detach_(), dim=-1)
        max_probs, targets_tu = torch.max(pseudo_label_tu, dim=-1)
        mask = max_probs.ge(args.threshold).float()


        #### Supervised Loss
        loss_x = criterion(out_2t, gt_labels_t) # + criterion(out_1s, gt_labels_s)

        ## Unsupervised Loss
        loss_u = (F.cross_entropy(out_2tu_strong, targets_tu, reduction='none') * mask).mean()

        loss = loss_x + 1.0 * loss_u


        loss.backward()
        optimizer_g.step()
        optimizer_f1.step()
        optimizer_f2.step()
        zero_grad_all()



        log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                    'Loss_x Classification: {:.6f} Loss_u Classification: {:.6f}\n'.format(args.source, args.target,
                                         step, lr, loss_x.data, loss_u.data)
        G.zero_grad()
        F1.zero_grad()
        F2.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val)
            G.train()
            F1.train()
            F2.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print('best acc test %f best acc val %f' % (best_acc_test,
                                                        best_acc))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f final %f \n' % (step,
                                                         best_acc_test,
                                                         best_acc))
            G.train()
            F1.train()
            F2.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(F2.state_dict(),
                           os.path.join(args.checkpath,
                                        "F2_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))


def test(loader):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output = F2(feat)
            output_all = np.r_[output_all, output.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()
