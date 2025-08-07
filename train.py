import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import joint_transforms_depth
from config import ViMirr_Depth_training_root, ViMirr_Depth_test_root
from dataset.crosspairwise_depth import CrossPairwiseImg
from misc import AvgMeter, check_mkdir
# from networks.TVSD import TVSD
# from networks.VMD_network import VMD_Network
from torch.optim.lr_scheduler import StepLR
import math
from util.loss.losses import lovasz_hinge, binary_xloss
import random
import torch.nn.functional as F
import numpy as np
# from apex import amp
import time
import argparse
import importlib
# from utils import backup_code
import utils_modify


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.deterministic = True
cudnn.benchmark = False

ckpt_path = './output/full_depth_vimirr'
# exp_name = 'VMD'

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='DVMDNet', help='exp name')
parser.add_argument('--model', type=str, default='DVMDNet', help='model name')
parser.add_argument('--gpu', type=str, default='0', help='used gpu id')
# parser.add_argument('--gpu', type=str, default='0,1', help='used gpu id')
# parser.add_argument('--batchsize', type=int, default=2, help='train batch')
parser.add_argument('--batchsize', type=int, default=2, help='train batch')
parser.add_argument('--bestonly', action="store_true", help='only best model')

cmd_args = parser.parse_args()
exp_name = cmd_args.exp
model_name = cmd_args.model
gpu_ids = cmd_args.gpu
train_batch_size = cmd_args.batchsize

VMD_file = importlib.import_module('networks.DVMDNet.' + model_name)
VMD_Network = VMD_file.VMD_Network
# print(torch.__version__)


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
# print(torch.cuda.device_count())

args = {
    # 'exp_name': exp_name,
    'max_epoch': 15,
    'train_batch_size': cmd_args.batchsize,
    'last_iter': 0,
    'finetune_lr': 1e-5,
    # 'finetune_lr': 6e-5,
    # 'scratch_lr': 6e-4,
    'scratch_lr': 1e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    # 'scale': 256,
    'scale': 512,
    'multi-scale': None,
    # 'gpu': '4,5',
    'gpu': '0,1',
    # 'multi-GPUs': True,
    'fp16': False,
    'warm_up_epochs': 3,
    'seed': 1234
}

# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

# multi-GPUs training
if len(gpu_ids.split(',')) > 1:
    # print(1)
    batch_size = train_batch_size * len(gpu_ids.split(','))
    # print("batch_size" + str(batch_size))
# single-GPU training
else:
    torch.cuda.set_device(0)
    batch_size = train_batch_size

joint_transform = joint_transforms_depth.Compose([
    joint_transforms_depth.Resize((args['scale'], args['scale'])),
    joint_transforms_depth.RandomHorizontallyFlip()
])
val_joint_transform = joint_transforms_depth.Compose([
    joint_transforms_depth.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

print('=====>Dataset loading<======')
training_root = [ViMirr_Depth_training_root] # training_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.
train_set = CrossPairwiseImg(training_root, joint_transform, img_transform, depth_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=batch_size,  drop_last=True, num_workers=4, shuffle=True)

val_set = CrossPairwiseImg([ViMirr_Depth_test_root], val_joint_transform, img_transform, depth_transform, target_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, shuffle=False)

print("max epoch:{}".format(args['max_epoch']))

# ce_loss = nn.CrossEntropyLoss()
segmentation_loss = binary_xloss
lovasz_hinge = lovasz_hinge

exp_time = datetime.datetime.now()
log_dir_path = os.path.join(ckpt_path, exp_name, str(exp_time))
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
check_mkdir(log_dir_path)
log_path = os.path.join(ckpt_path, exp_name, str(exp_time), 'train_log.txt')
val_log_path = os.path.join(ckpt_path, exp_name, str(exp_time), 'val_log.txt')
train_writer = SummaryWriter(log_dir=os.path.join(log_dir_path, 'train'), comment='train')


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def main():
    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if len(gpu_ids.split(',')) > 1:
        net = torch.nn.DataParallel(VMD_Network()).cuda().train()
        # for name, param in net.named_parameters():
        #     if 'backbone' in name:
        #         print(name)
        # net = net.apply(freeze_bn) # freeze BN
        params = [
            {"params": net.module.segformer.parameters(), "lr": args['finetune_lr']},

            {"params": net.module.aspp.parameters(), "lr": args['scratch_lr']},

            {"params": net.module.ra_attention_low.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.ra_attention_high.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.project.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.final_pre.parameters(), "lr": args['scratch_lr']},

        ]
    # single-GPU training
    else:
        net = VMD_Network().cuda().train()

        params = [
            {"params": net.segformer.parameters(), "lr": args['finetune_lr']},

            {"params": net.depth_conv0.parameters(), "lr": args['scratch_lr']},
            {"params": net.depth_conv1.parameters(), "lr": args['scratch_lr']},
            {"params": net.depth_conv2.parameters(), "lr": args['scratch_lr']},
            {"params": net.depth_conv3.parameters(), "lr": args['scratch_lr']},
            {"params": net.depth_conv4.parameters(), "lr": args['scratch_lr']},

            {"params": net.rgb_aspp.parameters(), "lr": args['scratch_lr']},
            {"params": net.depth_aspp.parameters(), "lr": args['scratch_lr']},

            {"params": net.ra_attention_spatial_high.parameters(), "lr": args['scratch_lr']},
            {"params": net.ra_attention_spatial_low.parameters(), "lr": args['scratch_lr']},

            {"params": net.ra_attention_low.parameters(), "lr": args['scratch_lr']},
            {"params": net.ra_attention_high.parameters(), "lr": args['scratch_lr']},
            {"params": net.project.parameters(), "lr": args['scratch_lr']},
            {"params": net.ffm.parameters(), "lr": args['scratch_lr']},
            {"params": net.final_pre.parameters(), "lr": args['scratch_lr']},
            {"params": net.query_pre.parameters(), "lr": args['scratch_lr']},

            {"params": net.exemplar_dm2.parameters(), "lr": args['scratch_lr']},
            {"params": net.query_dm2.parameters(), "lr": args['scratch_lr']},
            {"params": net.other_dm2.parameters(), "lr": args['scratch_lr']},
            {"params": net.exemplar_dm1.parameters(), "lr": args['scratch_lr']},
            {"params": net.query_dm1.parameters(), "lr": args['scratch_lr']},
            {"params": net.other_dm1.parameters(), "lr": args['scratch_lr']},

            {"params": net.final_examplar.parameters(), "lr": args['scratch_lr']},
            {"params": net.final_query.parameters(), "lr": args['scratch_lr']},
            {"params": net.final_other.parameters(), "lr": args['scratch_lr']},

        ]


    optimizer = optim.AdamW(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    open(log_path, 'w').write(str(args) + '\n\n')
    if args['fp16']:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    model_total_params = sum(p.numel() for p in net.parameters())
    model_grad_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    print('model: #params={}'.format(utils_modify.compute_num_params(net, text=True)))

    start = time.time()
    train(net, optimizer)
    end = time.time()
    print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
    open(log_path, 'a').write("Total Training Time: " + str(datetime.timedelta(seconds=int(end - start))) + '\n')




def train(net, optimizer):
    curr_epoch = 1
    curr_iter = 1
    start = 0
    best_mae = 100.0

    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record5, loss_record6, loss_record7, loss_record8 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record9, loss_record10, loss_record11, loss_record12 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record13, loss_record14, loss_record15, loss_record16 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record17, loss_record18, loss_record19, loss_record20 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader))

        train_writer.add_scalar('lr/parameter_group[0]', optimizer.param_groups[0]['lr'], curr_epoch)
        train_writer.add_scalar('lr/parameter_group[1]', optimizer.param_groups[1]['lr'], curr_epoch)
        train_writer.add_scalar('lr/parameter_group[2]', optimizer.param_groups[2]['lr'], curr_epoch)


        for i, sample in enumerate(train_iterator):

            exemplar, exemplar_depth, exemplar_gt = sample['exemplar'].cuda(), sample['exemplar_depth'].cuda(), sample['exemplar_gt'].cuda(),
            query, query_depth, query_gt = sample['query'].cuda(), sample['query_depth'].cuda(), sample['query_gt'].cuda()
            other, other_depth, other_gt = sample['other'].cuda(), sample['other_depth'].cuda(), sample['other_gt'].cuda()
            print(other_depth.shape)
            optimizer.zero_grad()

            exemplar_pre, query_pre1, query_pre2, query_pre3, other_pre, final_examplar, final_query, final_other = net(exemplar, query, other, exemplar_depth, query_depth, other_depth)

            exemplar_bce_loss = segmentation_loss(exemplar_pre, exemplar_gt)
            query_pre1_bce_loss = segmentation_loss(query_pre1, query_gt)
            query_pre2_bce_loss = segmentation_loss(query_pre2, query_gt)

            query_pre3_bce_loss = segmentation_loss(query_pre3, query_gt)
            other_bce_loss = segmentation_loss(other_pre, other_gt)
            final_examplar_bce_loss = segmentation_loss(final_examplar, exemplar_gt)
            final_query_bce_loss = segmentation_loss(final_query, query_gt)
            final_other_bce_loss = segmentation_loss(final_other, query_gt)

            exemplar_hinge_loss = lovasz_hinge(exemplar_pre, exemplar_gt)
            query_pre1_hinge_loss = lovasz_hinge(query_pre1, query_gt)
            query_pre2_hinge_loss = lovasz_hinge(query_pre2, query_gt)

            query_pre3_hinge_loss = lovasz_hinge(query_pre3, query_gt)
            other_hinge_loss = lovasz_hinge(other_pre, other_gt)
            final_examplar_hinge_loss = lovasz_hinge(final_examplar, exemplar_gt)
            final_query_hinge_loss = lovasz_hinge(final_query, query_gt)
            final_other_hinge_loss = lovasz_hinge(final_other, other_gt)


            loss_seg = (exemplar_bce_loss + query_pre1_bce_loss + query_pre2_bce_loss + query_pre3_bce_loss +
                        other_bce_loss + final_examplar_bce_loss + final_query_bce_loss + final_other_bce_loss +
                        exemplar_hinge_loss + query_pre1_hinge_loss + query_pre2_hinge_loss + query_pre3_hinge_loss +
                        other_hinge_loss + final_examplar_hinge_loss + final_query_hinge_loss + final_other_hinge_loss)

            loss = loss_seg

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()  # change gradient

            loss_record1.update(loss.item(), batch_size)
            loss_record2.update(exemplar_bce_loss.item(), batch_size)
            loss_record3.update(query_pre1_bce_loss.item(), batch_size)
            loss_record4.update(query_pre2_bce_loss.item(), batch_size)
            loss_record5.update(query_pre3_bce_loss.item(), batch_size)
            loss_record6.update(other_bce_loss.item(), batch_size)
            loss_record7.update(final_examplar_bce_loss.item(), batch_size)
            loss_record8.update(final_query_bce_loss.item(), batch_size)
            loss_record9.update(final_other_bce_loss.item(), batch_size)

            loss_record10.update(exemplar_hinge_loss.item(), batch_size)
            loss_record11.update(query_pre1_hinge_loss.item(), batch_size)
            loss_record12.update(query_pre2_hinge_loss.item(), batch_size)
            loss_record13.update(query_pre3_hinge_loss.item(), batch_size)
            loss_record14.update(other_hinge_loss.item(), batch_size)
            loss_record15.update(final_examplar_hinge_loss.item(), batch_size)
            loss_record16.update(final_query_hinge_loss.item(), batch_size)
            loss_record17.update(final_other_hinge_loss.item(), batch_size)

            train_writer.add_scalar('loss/total_loss', loss_record1.avg, curr_iter)
            train_writer.add_scalar('loss/exemplar_bce_loss', loss_record2.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre1_bce_loss', loss_record3.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre2_bce_loss', loss_record4.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre3_bce_loss', loss_record5.avg, curr_iter)
            train_writer.add_scalar('loss/other_bce_loss', loss_record6.avg, curr_iter)
            train_writer.add_scalar('loss/final_examplar_bce_loss', loss_record7.avg, curr_iter)
            train_writer.add_scalar('loss/final_query_bce_loss', loss_record8.avg, curr_iter)
            train_writer.add_scalar('loss/final_other_bce_loss', loss_record9.avg, curr_iter)

            train_writer.add_scalar('loss/exemplar_hinge_loss', loss_record10.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre1_hinge_loss', loss_record11.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre2_hinge_loss', loss_record12.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre3_hinge_loss', loss_record13.avg, curr_iter)
            train_writer.add_scalar('loss/other_hinge_loss', loss_record14.avg, curr_iter)
            train_writer.add_scalar('loss/final_examplar_hinge_loss', loss_record15.avg, curr_iter)
            train_writer.add_scalar('loss/final_query_hinge_loss', loss_record16.avg, curr_iter)
            train_writer.add_scalar('loss/final_other_hinge_loss', loss_record17.avg, curr_iter)

            logged_images = {
                "images/clip_1": (reverse_normalize(exemplar[0]).cpu().numpy() * 255).astype(np.uint8),
                "images/clip_2": (reverse_normalize(query[0]).cpu().numpy() * 255).astype(np.uint8),
                "images/clip_3": (reverse_normalize(other[0]).cpu().numpy() * 255).astype(np.uint8),
                "preds/clip_1": (final_examplar[0] > 0.5).to(torch.int8),
                "preds/clip_2": (final_query[0] > 0.5).to(torch.int8),
                "preds/clip_3": (final_other[0] > 0.5).to(torch.int8),
                "labels/clip_1": exemplar_gt[0].unsqueeze(0) if len(exemplar_gt[0].size()) == 2 else exemplar_gt[0],
                "labels/clip_2": query_gt[0].unsqueeze(0) if len(query_gt[0].size()) == 2 else query_gt[0],
                "labels/clip_3": other_gt[0].unsqueeze(0) if len(other_gt[0].size()) == 2 else other_gt[0],
                "depth/clip_1": exemplar_depth[0].unsqueeze(0) if len(exemplar_depth[0].size()) == 2 else
                exemplar_depth[0],
                "depth/clip_2": query_depth[0].unsqueeze(0) if len(query_depth[0].size()) == 2 else query_depth[0],
                "depth/clip_3": other_depth[0].unsqueeze(0) if len(other_depth[0].size()) == 2 else other_depth[0],
            }

            for image_name, image in logged_images.items():

                train_writer.add_image("{}".format(image_name), image, curr_iter)  # self.current_epoch)

            curr_iter += 1
            log = (
                      "epochs:%d, iter: %d, loss: %f5, exemplar_bce_loss: %f5, query_pre1_bce_loss: %f5, query_pre2_bce_loss: %f5,  "
                      "query_pre3_bce_loss: %f5, other_bce_loss: %f5, final_examplar_bce_loss: %f5, final_query_bce_loss: %f5, "
                      "final_other_bce_loss: %f5, "
                      "exemplar_hinge_loss:%f5, query_pre1_hinge_loss:%f5, query_pre2_hinge_loss:%f5, query_final_hinge_loss: %f5, "
                      "other_hinge_loss:%f5, final_examplar_hinge_loss:%f5, final_query_hinge_loss:%f5, final_other_hinge_loss:%f5, "
                      "lr: %f8") % \
                  (curr_epoch, curr_iter, loss_record1.avg, loss_record2.avg, loss_record3.avg, loss_record4.avg,
                   loss_record5.avg, loss_record6.avg, loss_record7.avg, loss_record8.avg, loss_record9.avg,
                   loss_record10.avg, loss_record11.avg,  loss_record12.avg, loss_record13.avg, loss_record14.avg,
                   loss_record15.avg, loss_record16.avg, loss_record17.avg,
                   optimizer.param_groups[0]['lr'])

            if (curr_iter-1) % 20 == 0:
                elapsed = (time.perf_counter() - start)
                start = time.perf_counter()
                log_time = log + ' [time {}]'.format(elapsed)
                print(log_time)
                # train_iterator.set_description(log_time)
            open(log_path, 'a').write(log + '\n')

        if curr_epoch % 1 == 0 and not cmd_args.bestonly:
            # if args['multi-GPUs']:
            if len(gpu_ids.split(',')) > 1:
                # torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                if args['fp16']:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, str(exp_time), f'{curr_epoch}.pth'))
            else:

                if args['fp16']:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, str(exp_time), f'{curr_epoch}.pth'))


        current_mae = val(net, curr_epoch)

        net.train() # val -> train
        if current_mae < best_mae:
            best_mae = current_mae
            if len(gpu_ids.split(',')) > 1:
                if args['fp16']:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
            else:
                if args['fp16']:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
            torch.save(checkpoint, os.path.join(ckpt_path, exp_name, str(exp_time), 'best_mae.pth'))



        if curr_epoch > args['max_epoch']:
            # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
            return

        curr_epoch += 1


def val(net, epoch):
    mae_record = AvgMeter()
    net.eval()
    with torch.no_grad():
        val_iterator = tqdm(val_loader)
        for i, sample in enumerate(val_iterator):
            exemplar, exemplar_depth, exemplar_gt = sample['exemplar'].cuda(), sample['exemplar_depth'].cuda(), sample['exemplar_gt'].cuda(),
            query, query_depth, query_gt = sample['query'].cuda(), sample['query_depth'].cuda(), sample['query_gt'].cuda()
            other, other_depth, other_gt = sample['other'].cuda(), sample['other_depth'].cuda(), sample['other_gt'].cuda()

            _, query_final, _ = net(exemplar, query, other, exemplar_depth, query_depth, other_depth)


            res = (query_final.data > 0).to(torch.float32).squeeze(0)

            mae = torch.mean(torch.abs(res - query_gt.squeeze(0)))

            batch_size = query.size(0)
            mae_record.update(mae.item(), batch_size)

        train_writer.add_scalar("val/mae", mae_record.avg, epoch)
        log = "val: iter: %d, mae: %f5" % (epoch, mae_record.avg)
        print(log)
        open(val_log_path, 'a').write(log + '\n')
        return mae_record.avg

def reverse_normalize(normalized_image):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    inv_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    inv_tensor = inv_normalize(normalized_image)
    return inv_tensor

if __name__ == '__main__':
    main()
