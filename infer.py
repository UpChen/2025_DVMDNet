import numpy as np
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import joint_transforms_depth
from config import ViMirr_Depth_test_root
from misc import check_mkdir
# from networks.TVSD import TVSD
from networks.DVMDNet.DVMDNet import DVMD_Network
from dataset.crosspairwise_depth import listdirs_only, CrossPairwiseImg
import argparse
from tqdm import tqdm
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = {
    'scale': 512,
    'test_adjacent': 1,
    'input_folder': 'JPEGImages',
    'label_folder': 'Annotations'
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
depth_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
])
target_transform = transforms.ToTensor()

val_joint_transform = joint_transforms_depth.Compose([
    joint_transforms_depth.Resize((args['scale'], args['scale']))
])

root = ViMirr_Depth_test_root[0]

to_pil = transforms.ToPILImage()

val_set = CrossPairwiseImg([ViMirr_Depth_test_root], val_joint_transform, img_transform, depth_transform, target_transform)
val_loader = DataLoader(val_set, batch_size=1, num_workers=1, shuffle=False)


def main():
    net = DVMD_Network().cuda()

    
    checkpoint = ''
    save_dir = ""
    check_point = torch.load(checkpoint)
    net.load_state_dict(check_point['model'])

    net.eval()
    with torch.no_grad():
        old_temp = ''
        val_iterator = tqdm(val_loader)
        for i, sample in enumerate(val_iterator):
            exemplar, exemplar_depth, exemplar_gt = sample['exemplar'].cuda(), sample['exemplar_depth'].cuda(), sample[
                'exemplar_gt'].cuda(),
            query, query_depth, query_gt = sample['query'].cuda(), sample['query_depth'].cuda(), sample[
                'query_gt'].cuda()
            other, other_depth, other_gt = sample['other'].cuda(), sample['other_depth'].cuda(), sample[
                'other_gt'].cuda()
            video_name = sample['video_name'][0]

            if old_temp == video_name:

                query_index = query_index + 1
            else:

                query_index = 0

            _, query_final, _ = net(exemplar, query, other, exemplar_depth, query_depth, other_depth)
            res = (query_final.data > 0).to(torch.float32).squeeze(0)

            query_index1 = str(query_index).zfill(5)

            first_image = np.array(Image.open(root + '/JPEGImages/' + str(video_name) + '/' + query_index1 + '.jpg'))

            h, w, _ = first_image.shape

            prediction = np.array(
                transforms.Resize((h, w))(to_pil(res.cpu())))

            check_mkdir(os.path.join(save_dir, str(video_name)))

            query_save_name = f"{query_index1}.png"

            Image.fromarray(prediction).save(os.path.join(save_dir, str(video_name), query_save_name))

            old_temp = video_name



def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]


def getAdjacentIndex(current_index, start_index, video_length, adjacent_length):
    if current_index + adjacent_length < start_index + video_length:
        query_index_list = [current_index+i+1 for i in range(adjacent_length)]
    else:
        query_index_list = [current_index-i-1 for i in range(adjacent_length)]
    return query_index_list

if __name__ == '__main__':
    main()
