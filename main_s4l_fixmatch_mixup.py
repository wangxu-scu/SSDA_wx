from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset_s4l_fixmatch_mixup
import torch.nn.functional as F
import random
import time

# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='S4L_FIXMATCH',
                    choices=['S+T', 'ENT', 'MME', 'S4L_FIXMATCH'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
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


parser.add_argument('--threshold', type=float, default=0.95, metavar='THRE',
                    help='value of threshold')
parser.add_argument('--exp_name', type=str, default='2nd_3',
                    help='the name of the experiment')

args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset_s4l_fixmatch_mixup(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s_expname_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num, args.exp_name))

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)


def mixup_data(x_tu, x_st, y_tu, y_st, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    x_a = torch.cat((x_tu, x_st), 0)
    y_a = torch.cat((y_tu, y_st), 0)
    batch_size = x_st.size()[0]
    index = np.random.permutation(batch_size) + x_tu.size()[0]
    index = np.concatenate((np.arange(x_tu.size()[0]), index), 0)
    mixed_x = lam * x_a + (1 - lam) * x_a[index, :]
    # y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y_a, y_a[index]
    mixed_x = mixed_x[x_tu.size()[0] // 2:]
    y_a = y_a[x_tu.size()[0] // 2:]
    y_b = y_b[x_tu.size()[0] // 2:]
    return mixed_x, y_a, y_b, lam

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
## F2: Target-based Classifier
if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
    F2 = Predictor_deep(num_class=4,
                        inc=inc)
else:
    #### F1: Semantic Classifier; F2: Rotation Classifier
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
    F2 = Predictor(num_class=4, inc=inc,
                   temp=args.T)
weights_init(F1)
weights_init(F2)
lr = args.lr
G.cuda()
F1.cuda()
F2.cuda()


if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    F1.train()
    F2.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
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
    end = time.time()
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
        im_data_s = data_s[0].cuda()
        im_data_s = im_data_s.reshape(-1, im_data_s.shape[2], im_data_s.shape[3], im_data_s.shape[4])
        im_data_s_strong = data_s[1].cuda()
        gt_labels_s = data_s[2].cuda()
        gt_labels_s = torch.transpose(gt_labels_s, 1, 2)
        gt_labels_s = gt_labels_s.reshape(gt_labels_s.shape[0] * gt_labels_s.shape[1], gt_labels_s.shape[2])
        im_data_t = data_t[0].cuda()
        im_data_t = im_data_t.reshape(-1, im_data_t.shape[2], im_data_t.shape[3], im_data_t.shape[4])
        im_data_t_strong = data_t[1].cuda()
        gt_labels_t = data_t[2].cuda()
        gt_labels_t = torch.transpose(gt_labels_t, 1, 2)
        gt_labels_t = gt_labels_t.reshape(gt_labels_t.shape[0] * gt_labels_t.shape[1], gt_labels_t.shape[2])
        im_data_tu = data_t_unl[0].cuda()
        im_data_tu = im_data_tu.reshape(-1, im_data_tu.shape[2], im_data_tu.shape[3], im_data_tu.shape[4])
        im_data_tu_strong = data_t_unl[1].cuda()
        gt_labels_tu = data_t_unl[2].cuda()
        gt_labels_tu = torch.transpose(gt_labels_tu, 1, 2)
        gt_labels_tu = gt_labels_tu.reshape(gt_labels_tu.shape[0] * gt_labels_tu.shape[1], gt_labels_tu.shape[2])
        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t, im_data_tu), 0)
        zero_grad_all()

        output = G(data)
        output_s = output[:len(im_data_s)]
        output_t = output[len(im_data_s): len(im_data_s) + len(im_data_t)]
        output_tu = output[len(im_data_s) + len(im_data_t):]

        #### Supervised Loss for unrotated images
        output_s_no_rot = output_s.index_select(0, torch.arange(0, len(output_s), 4).cuda())
        output_t_no_rot = output_t.index_select(0, torch.arange(0, len(output_t), 4).cuda())
        gt_labels_s_cls = gt_labels_s[:, 0].index_select(0, torch.arange(0, len(output_s), 4).cuda())
        gt_labels_t_cls = gt_labels_t[:, 0].index_select(0, torch.arange(0, len(output_t), 4).cuda())
        logits_l_cls = F1(torch.cat((output_s_no_rot, output_t_no_rot), 0))
        target_l_cls = torch.cat((gt_labels_s_cls, gt_labels_t_cls), 0)

        loss_x = criterion(logits_l_cls, target_l_cls)

        ## Unsupervised Loss
        output_tu_no_rot = output_tu.index_select(0, torch.arange(0, len(output_tu), 4).cuda())
        logits_tu_weak = F1(output_tu_no_rot)
        pseudo_label_tu = torch.softmax(logits_tu_weak.detach_(), dim=-1)
        max_probs, targets_tu = torch.max(pseudo_label_tu, dim=-1)

        mask = max_probs.ge(args.threshold).float().repeat(3)

        x_tu = torch.cat((im_data_tu_strong, im_data_tu_strong), 0)
        x_st = torch.cat((im_data_s_strong, im_data_t_strong), 0)
        y_tu = torch.cat((targets_tu, targets_tu), 0)
        y_st = torch.cat((gt_labels_s_cls, gt_labels_t_cls), 0)

        mixed_x_strong, y_a, y_b, lam = mixup_data(x_tu, x_st, y_tu, y_st, alpha=1.0)
        logits_mix_strong = F1(G(mixed_x_strong))
        loss_u = lam * (F.cross_entropy(logits_mix_strong, y_a, reduction='none') * mask).mean() + \
                 (1 - lam) * (F.cross_entropy(logits_mix_strong, y_b, reduction='none') * mask).mean()

        ### Rotation Self-supervised Loss
        logits_ul_rot = F2(torch.cat((output_s, output_t, output_tu), 0))
        target_ul_rot = torch.cat((gt_labels_s[:, 1], gt_labels_t[:, 1], gt_labels_tu[:, 1]), 0)
        loss_rot = criterion(logits_ul_rot, target_ul_rot.cuda())

        ### Overall Loss
        loss = loss_x + loss_u + 0.7 * loss_rot

        loss.backward()
        optimizer_g.step()
        optimizer_f1.step()
        optimizer_f2.step()
        zero_grad_all()

        G.zero_grad()
        F1.zero_grad()
        F2.zero_grad()
        zero_grad_all()

        if step % args.log_interval == 0:
            time_elapse = time.time() - end
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'loss: {:.6f} loss_x: {:.6f} ' \
                        'loss_u {:.6f} loss_rot: {:.6f} time_elapse: {:.3f}s ' \
                        'Method {}\n'. \
                format(args.source, args.target,
                       step, lr, loss.data, loss_x.data,
                       loss_u.data, loss_rot.data, time_elapse,
                       args.method)
            end = time.time()
            print(log_train)
            with open(record_file, 'a') as f:
                f.write(log_train)
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
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
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
            im_data_t = data_t[0].cuda()
            gt_labels_t = data_t[1].cuda()
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.1f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()
