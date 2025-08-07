import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation

from .pacconv import PacConv2d


class DVMD_Network(nn.Module):
    def __init__(self, pretrained_path=None, num_classes=1, all_channel=256, all_dim=26 * 26, T=0.07):  # 473./8=60 416./8=52
        super(DVMD_Network, self).__init__()
        self.pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-ade-512-512", num_labels=1, ignore_mismatched_sizes=True)
        self.config = self.pretrained_model.config
        self.segformer = self.pretrained_model.segformer
        # self.decode_head = self.pretrained_model.decode_head

        aspp_dilate = [12, 24, 36]
        # aspp_dilate = [6, 12, 18]
        self.rgb_aspp = ASPP(512, aspp_dilate, 256)
        self.depth_aspp = ASPP(512, aspp_dilate, 256)

        # depth feature extraction
        self.depth_conv0 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv1 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv4 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.ra_attention_spatial_high = Relation_Attention_Diff(in_channels_x=256, in_channels_y=256)
        self.ra_attention_spatial_low = Relation_Attention_Diff(in_channels_x=128, in_channels_y=128)

        self.ra_attention_high = Relation_Attention(in_channels_x=256, in_channels_y=256)
        self.ra_attention_low = Relation_Attention(in_channels_x=128, in_channels_y=128)

        self.project = nn.Sequential(
            nn.Conv2d(128, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.final_pre = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self.ffm = FFM(304)

        self.query_pre = conv3x3(304, 1, bias=True)

        # delineating
        self.exemplar_dm2 = DM(512, 512)
        self.query_dm2 = Temporal_DM(512, 512, 1024)
        self.other_dm2 = DM(512, 512)

        self.exemplar_dm1 = high_level_DM(128, 128, 1024)
        self.query_dm1 = Temporal_high_level_DM(128, 128, 1024, 256)
        self.other_dm1 = high_level_DM(128, 128, 1024)

        #
        self.final_examplar = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

        self.final_query = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        self.final_other = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        initialize_weights(self.depth_conv0, self.depth_conv1, self.depth_conv2, self.depth_conv3, self.depth_conv4,
                           self.rgb_aspp, self.depth_aspp, self.ra_attention_spatial_low, self.ra_attention_spatial_high,
                           self.ra_attention_high, self.ra_attention_low, self.ffm, self.project, self.query_pre,
                           self.final_pre, self.exemplar_dm2, self.exemplar_dm1, self.query_dm2, self.query_dm1,
                           self.other_dm1, self.other_dm2, self.final_query, self.final_examplar, self.final_other)


    def forward(self, input1, input2, input3, input1_depth, input2_depth, input3_depth, labels = None, output_attentions = None, output_hidden_states = True, return_dict = None):
        input_size = input1.size()[2:]

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        exemplar_outputs = self.segformer(
            input1,
            output_attentions=True,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        query_outputs = self.segformer(
            input2,
            output_attentions=True,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        other_outputs = self.segformer(
            input3,
            output_attentions=True,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        exemplar_encoder_hidden_states = exemplar_outputs.hidden_states if return_dict else exemplar_outputs[1]
        query_encoder_hidden_states = query_outputs.hidden_states if return_dict else query_outputs[1]
        other_encoder_hidden_states = other_outputs.hidden_states if return_dict else other_outputs[1]

        depth_exemplar_conv0 = self.depth_conv0(input1_depth)
        depth_exemplar_conv1 = self.depth_conv1(depth_exemplar_conv0)
        depth_exemplar_conv2 = self.depth_conv2(depth_exemplar_conv1)
        depth_exemplar_conv3 = self.depth_conv3(depth_exemplar_conv2)
        depth_exemplar_conv4 = self.depth_conv4(depth_exemplar_conv3)
        depth_exemplar_conv5 = self.depth_aspp(depth_exemplar_conv4)

        depth_query_conv0 = self.depth_conv0(input2_depth)  # torch.Size([5, 8, 256, 256])
        depth_query_conv1 = self.depth_conv1(depth_query_conv0)  # torch.Size([5, 16, 128, 128])
        depth_query_conv2 = self.depth_conv2(depth_query_conv1)  # torch.Size([5, 32, 64, 64])
        depth_query_conv3 = self.depth_conv3(depth_query_conv2)  # torch.Size([5, 64, 32, 32])
        depth_query_conv4 = self.depth_conv4(depth_query_conv3)  # torch.Size([5, 128, 16, 16])
        depth_query_conv5 = self.depth_aspp(depth_query_conv4)  # torch.Size([5, 256, 16, 16])

        depth_other_conv0 = self.depth_conv0(input3_depth)  # torch.Size([5, 8, 256, 256])
        depth_other_conv1 = self.depth_conv1(depth_other_conv0)  # torch.Size([5, 16, 128, 128])
        depth_other_conv2 = self.depth_conv2(depth_other_conv1)  # torch.Size([5, 32, 64, 64])
        depth_other_conv3 = self.depth_conv3(depth_other_conv2)  # torch.Size([5, 64, 32, 32])
        depth_other_conv4 = self.depth_conv4(depth_other_conv3)  # torch.Size([5, 128, 16, 16])
        depth_other_conv5 = self.depth_aspp(depth_other_conv4)  # torch.Size([5, 256, 16, 16])

        low_rgb_exemplar = exemplar_encoder_hidden_states[1]
        low_rgb_query = query_encoder_hidden_states[1]
        low_rgb_other = other_encoder_hidden_states[1]

        rgb_exemplar4 = exemplar_encoder_hidden_states[-1]
        rgb_query4 = query_encoder_hidden_states[-1]
        rgb_other4 = other_encoder_hidden_states[-1]
        rgb_exemplar = self.rgb_aspp(exemplar_encoder_hidden_states[-1])
        rgb_query = self.rgb_aspp(query_encoder_hidden_states[-1])
        rgb_other = self.rgb_aspp(other_encoder_hidden_states[-1])

        depth_exemplar_conv1_small = F.interpolate(depth_exemplar_conv1, size=low_rgb_exemplar.shape[2:], mode='bilinear', align_corners=False)
        depth_query_conv1_small = F.interpolate(depth_query_conv1, size=low_rgb_query.shape[2:], mode='bilinear', align_corners=False)
        depth_other_conv1_small = F.interpolate(depth_other_conv1, size=low_rgb_other.shape[2:], mode='bilinear', align_corners=False)

        y1, y2 = self.ra_attention_spatial_low(low_rgb_exemplar, low_rgb_query, depth_exemplar_conv1_small, depth_query_conv1_small)

        y3, y4 = self.ra_attention_spatial_high(rgb_exemplar, rgb_query, depth_exemplar_conv5, depth_query_conv5)
        y3 = F.interpolate(y3, size=low_rgb_exemplar.shape[2:], mode='bilinear', align_corners=False)
        y4 = F.interpolate(y4, size=low_rgb_query.shape[2:], mode='bilinear', align_corners=False)
        fuse_y1_y3 = torch.cat([y3, self.project(y1)], dim=1)
        fuse_y2_y4 = torch.cat([y4, self.project(y2)], dim=1)

        exemplar_pre = self.final_pre(fuse_y1_y3)
        query_pre1 = self.final_pre(fuse_y2_y4)

        exemplar_pre = F.upsample(exemplar_pre, input_size, mode='bilinear',
                                  align_corners=False)  # upsample to the size of input image, scale=8
        query_pre1 = F.upsample(query_pre1, input_size, mode='bilinear',
                                align_corners=False)  # upsample to the size of input image, scale=8

        low_query, low_other = self.ra_attention_low(low_rgb_query, low_rgb_other, depth_query_conv1_small, depth_other_conv1_small)

        x1, x2 = self.ra_attention_high(rgb_query, rgb_other, depth_query_conv5, depth_other_conv5)
        x1 = F.interpolate(x1, size=low_query.shape[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=low_other.shape[2:], mode='bilinear', align_corners=False)
        fuse_query = torch.cat([x1, self.project(low_query)], dim=1)
        fuse_other = torch.cat([x2, self.project(low_other)], dim=1)
        query_pre2 = self.final_pre(fuse_query)

        final_fuse_query = self.ffm(fuse_y2_y4, fuse_query)

        query_pre3 = self.query_pre(final_fuse_query)
        other_pre = self.final_pre(fuse_other)

        query_pre2 = F.upsample(query_pre2, input_size, mode='bilinear',
                                align_corners=False)  # upsample to the size of input image, scale=8
        query_pre3 = F.upsample(query_pre3, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        other_pre = F.upsample(other_pre, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8

        examplar_small = F.interpolate(exemplar_pre, size=rgb_exemplar4.shape[2:], mode='bilinear', align_corners=False)
        query_small = F.interpolate(query_pre3, size=rgb_query4.shape[2:], mode='bilinear', align_corners=False)

        examplar_big = F.interpolate(exemplar_pre, size=depth_exemplar_conv1_small.shape[2:], mode='bilinear',
                                     align_corners=False)
        query_big = F.interpolate(query_pre3, size=depth_query_conv1_small.shape[2:], mode='bilinear',
                                  align_corners=False)

        sigmoid_examplar_small = torch.sigmoid(examplar_small)
        sigmoid_query_small = torch.sigmoid(query_small)

        sigmoid_examplar_big = torch.sigmoid(examplar_big)
        sigmoid_query_big = torch.sigmoid(query_big)

        # TODO: use content discontinuity attention to extract mirror contrast features
        # delineating
        exemplar_dm2 = self.exemplar_dm2(sigmoid_examplar_small * rgb_exemplar4, sigmoid_examplar_small * depth_exemplar_conv4)
        other_dm2 = self.other_dm2(rgb_other4, depth_other_conv4)
        query_dm2 = self.query_dm2(sigmoid_query_small * rgb_query4, sigmoid_query_small * depth_query_conv4, exemplar_dm2, other_dm2)

        exemplar_dm1 = self.exemplar_dm1(sigmoid_examplar_big * low_rgb_exemplar, sigmoid_examplar_big * depth_exemplar_conv1_small, exemplar_dm2)
        other_dm1 = self.other_dm1(low_rgb_other, depth_other_conv1_small, other_dm2)
        query_dm1 = self.query_dm1(sigmoid_query_big * low_rgb_query, sigmoid_query_big * depth_query_conv1_small, query_dm2, exemplar_dm1, other_dm1)

        final_examplar = self.final_examplar(exemplar_dm1)

        final_query = self.final_query(query_dm1)

        final_other = self.final_other(other_dm1)

        final_examplar = F.upsample(final_examplar, input_size, mode='bilinear',
                                    align_corners=False)  # upsample to the size of input image, scale=8
        final_query = F.upsample(final_query, input_size, mode='bilinear',
                                 align_corners=False)  # upsample to the size of input image, scale=8

        final_other = F.upsample(final_other, input_size, mode='bilinear',
                                 align_corners=False)  # upsample to the size of input image, scale=8

        if self.training:
            return exemplar_pre, query_pre1, query_pre2, query_pre3, other_pre, final_examplar, final_query, final_other
        else:
            return final_examplar, final_query, final_other


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


class DM(nn.Module):
    """ delineating module """

    def __init__(self, in_dim_x, in_dim_y):
        super(DM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2


        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

    def forward(self, x, y):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H1 X W1)
                y : input depth feature maps (B X C2 X H1 X W1)
                z : input higher-level feature maps (B X C3 X H2 X W2)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H1 X W1)
        """

        fusion1 = self.fusion1(torch.cat((x, y), 1))

        local_main = self.local_main(fusion1)
        context_main = self.context_main(fusion1)
        contrast_main = self.relu_main(self.bn_main(local_main - context_main))

        local_rgb = self.local_rgb(x)
        context_rgb = self.context_rgb(x)
        contrast_rgb = self.relu_rgb(self.bn_rgb(local_rgb - context_rgb))

        local_depth = self.local_depth(y)
        context_depth = self.context_depth(y)
        contrast_depth = self.relu_depth(self.bn_depth(local_depth - context_depth))

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        return fusion2


class high_level_DM(nn.Module):
    """ delineating module """

    def __init__(self, in_dim_x, in_dim_y, in_dim_z):
        super(high_level_DM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2
        self.in_dim_z = in_dim_z

        self.up_main = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=4))
        self.up_rgb = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_x, 3, 1, 1),
                                    nn.BatchNorm2d(self.in_dim_x), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=4))
        self.up_depth = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_y, 3, 1, 1),
                                      nn.BatchNorm2d(self.in_dim_y), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=4))

        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

    def forward(self, x, y, z):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H1 X W1)
                y : input depth feature maps (B X C2 X H1 X W1)
                z : input higher-level feature maps (B X C3 X H2 X W2)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H1 X W1)
        """
        up_main = self.up_main(z)

        fusion1 = self.fusion1(torch.cat((x, y), 1))
        feature_main = fusion1 + up_main
        local_main = self.local_main(feature_main)
        context_main = self.context_main(feature_main)
        contrast_main = self.relu_main(self.bn_main(local_main - context_main))

        up_rgb = self.up_rgb(z)
        feature_rgb = x + up_rgb
        local_rgb = self.local_rgb(feature_rgb)
        context_rgb = self.context_rgb(feature_rgb)
        contrast_rgb = self.relu_rgb(self.bn_rgb(local_rgb - context_rgb))

        up_depth = self.up_depth(z)
        feature_depth = y + up_depth
        local_depth = self.local_depth(feature_depth)
        context_depth = self.context_depth(feature_depth)
        contrast_depth = self.relu_depth(self.bn_depth(local_depth - context_depth))

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        return fusion2


class connect_DM(nn.Module):
    """ delineating module """

    def __init__(self, in_dim_x, in_dim_y, in_dim_z):
        super(connect_DM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2
        self.in_dim_z = in_dim_z

        self.up_main = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.up_rgb = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_x, 3, 1, 1),
                                    nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.up_depth = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_y, 3, 1, 1),
                                      nn.BatchNorm2d(self.in_dim_y), nn.ReLU())

        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

    def forward(self, x, y, z):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H1 X W1)
                y : input depth feature maps (B X C2 X H1 X W1)
                z : input higher-level feature maps (B X C3 X H2 X W2)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H1 X W1)
        """
        up_main = self.up_main(z)

        fusion1 = self.fusion1(torch.cat((x, y), 1))

        feature_main = fusion1 + up_main
        local_main = self.local_main(feature_main)
        context_main = self.context_main(feature_main)
        contrast_main = self.relu_main(self.bn_main(local_main - context_main))

        up_rgb = self.up_rgb(z)
        feature_rgb = x + up_rgb
        local_rgb = self.local_rgb(feature_rgb)
        context_rgb = self.context_rgb(feature_rgb)
        contrast_rgb = self.relu_rgb(self.bn_rgb(local_rgb - context_rgb))

        up_depth = self.up_depth(z)
        feature_depth = y + up_depth
        local_depth = self.local_depth(feature_depth)
        context_depth = self.context_depth(feature_depth)
        contrast_depth = self.relu_depth(self.bn_depth(local_depth - context_depth))

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        return fusion2

class connect_temporal_DM(nn.Module):
    """ delineating module """

    def __init__(self, in_dim_x, in_dim_y, in_dim_z, in_dim_t):
        super(connect_temporal_DM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2
        self.in_dim_z = in_dim_z
        self.in_dim_t = in_dim_t

        self.up_main = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.up_rgb = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_x, 3, 1, 1),
                                    nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.up_depth = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_y, 3, 1, 1),
                                      nn.BatchNorm2d(self.in_dim_y), nn.ReLU())

        self.previous_rgb = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_x, 3, 1, 1),
                                          nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.other_rgb = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_x, 3, 1, 1),
                                          nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.previous_depth = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_y, 3, 1, 1),
                                            nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.other_depth = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_y, 3, 1, 1),
                                            nn.BatchNorm2d(self.in_dim_y), nn.ReLU())

        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

    def forward(self, x, y, z, t_exemplar, t_other):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H1 X W1)
                y : input depth feature maps (B X C2 X H1 X W1)
                z : input higher-level feature maps (B X C3 X H2 X W2)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H1 X W1)
        """
        up_main = self.up_main(z)

        fusion1 = self.fusion1(torch.cat((x, y), 1))

        feature_main = fusion1 + up_main + t_exemplar + t_other
        local_main = self.local_main(feature_main)
        context_main = self.context_main(feature_main)
        contrast_main = self.relu_main(self.bn_main(local_main - context_main))

        up_rgb = self.up_rgb(z)
        previous_rgb = self.previous_rgb(t_exemplar)
        other_rgb = self.other_rgb(t_other)
        feature_rgb = x + up_rgb + previous_rgb + other_rgb
        local_rgb = self.local_rgb(feature_rgb)
        context_rgb = self.context_rgb(feature_rgb)
        contrast_rgb = self.relu_rgb(self.bn_rgb(local_rgb - context_rgb))

        up_depth = self.up_depth(z)
        previous_depth = self.previous_depth(t_exemplar)
        other_depth = self.other_depth(t_other)
        feature_depth = y + up_depth + previous_depth + other_depth
        local_depth = self.local_depth(feature_depth)
        context_depth = self.context_depth(feature_depth)
        contrast_depth = self.relu_depth(self.bn_depth(local_depth - context_depth))

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        return fusion2



class Temporal_DM(nn.Module):
    """ delineating module """

    def __init__(self, in_dim_x, in_dim_y, in_dim_t):
        super(Temporal_DM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2
        self.in_dim_t = in_dim_t

        # self.up_main = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_xy, 3, 1, 1),
        #                              nn.BatchNorm2d(self.in_dim_xy), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.previous_rgb = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_x, 3, 1, 1),
                                    nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.other_rgb = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_x, 3, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.previous_depth = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_y, 3, 1, 1),
                                      nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.other_depth = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_y, 3, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())

        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

    def forward(self, x, y, t_exemplar, t_other):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H1 X W1)
                y : input depth feature maps (B X C2 X H1 X W1)
                z : input higher-level feature maps (B X C3 X H2 X W2)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H1 X W1)
        """

        fusion1 = self.fusion1(torch.cat((x, y), 1))
        # print(fusion1.shape)
        feature_main = fusion1 + t_exemplar + t_other
        local_main = self.local_main(feature_main)
        context_main = self.context_main(feature_main)
        contrast_main = self.relu_main(self.bn_main(local_main - context_main))

        previous_rgb = self.previous_rgb(t_exemplar)
        other_rgb = self.other_rgb(t_other)
        feature_rgb = x + previous_rgb + other_rgb
        local_rgb = self.local_rgb(feature_rgb)
        context_rgb = self.context_rgb(feature_rgb)
        contrast_rgb = self.relu_rgb(self.bn_rgb(local_rgb - context_rgb))

        previous_depth = self.previous_depth(t_exemplar)
        other_depth = self.other_depth(t_other)
        feature_depth = y + previous_depth + other_depth
        local_depth = self.local_depth(feature_depth)
        context_depth = self.context_depth(feature_depth)
        contrast_depth = self.relu_depth(self.bn_depth(local_depth - context_depth))

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        return fusion2


class Temporal_high_level_DM(nn.Module):
    """ delineating module """

    def __init__(self, in_dim_x, in_dim_y, in_dim_z, in_dim_t):
        super(Temporal_high_level_DM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2
        self.in_dim_z = in_dim_z
        self.in_dim_t = in_dim_t

        self.up_main = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=4))
        self.up_rgb = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_x, 3, 1, 1),
                                    nn.BatchNorm2d(self.in_dim_x), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=4))
        self.up_depth = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_y, 3, 1, 1),
                                      nn.BatchNorm2d(self.in_dim_y), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=4))

        self.previous_rgb = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_x, 3, 1, 1),
                                    nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.other_rgb = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_x, 3, 1, 1),
                                          nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.previous_depth = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_y, 3, 1, 1),
                                      nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.other_depth = nn.Sequential(nn.Conv2d(self.in_dim_t, self.in_dim_y, 3, 1, 1),
                                            nn.BatchNorm2d(self.in_dim_y), nn.ReLU())

        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

    def forward(self, x, y, z, t_exemplar, t_other):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H1 X W1)
                y : input depth feature maps (B X C2 X H1 X W1)
                z : input higher-level feature maps (B X C3 X H2 X W2)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H1 X W1)
        """
        up_main = self.up_main(z)
        fusion1 = self.fusion1(torch.cat((x, y), 1))
        # print(fusion1.shape)
        feature_main = fusion1 + t_exemplar + t_other + up_main
        local_main = self.local_main(feature_main)
        context_main = self.context_main(feature_main)
        contrast_main = self.relu_main(self.bn_main(local_main - context_main))

        up_rgb = self.up_rgb(z)
        previous_rgb = self.previous_rgb(t_exemplar)
        other_rgb = self.other_rgb(t_other)
        feature_rgb = x + previous_rgb + up_rgb + other_rgb
        local_rgb = self.local_rgb(feature_rgb)
        context_rgb = self.context_rgb(feature_rgb)
        contrast_rgb = self.relu_rgb(self.bn_rgb(local_rgb - context_rgb))

        up_depth = self.up_depth(z)
        previous_depth = self.previous_depth(t_exemplar)
        other_depth = self.other_depth(t_other)
        feature_depth = y + previous_depth + other_depth + up_depth
        local_depth = self.local_depth(feature_depth)
        context_depth = self.context_depth(feature_depth)
        contrast_depth = self.relu_depth(self.bn_depth(local_depth - context_depth))

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        return fusion2


# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
        stride=stride, bias=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        # out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def corr_fun(Kernel_tmp, Feature):
    size = Kernel_tmp.size()
    CORR = []
    Kernel = []
    # print(len(Feature))  # batch_size
    for i in range(len(Feature)):
        ker = Kernel_tmp[i:i + 1]
        fea = Feature[i:i + 1]
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
        ker = ker.unsqueeze(2).unsqueeze(3)  # torch.Size([1024, 128, 1, 1])

        co = F.conv2d(fea, ker.contiguous())
        CORR.append(co)
        ker = ker.unsqueeze(0)
        Kernel.append(ker)
    corr = torch.cat(CORR, 0)
    Kernel = torch.cat(Kernel, 0)
    return corr, Kernel


class CorrelationLayer(nn.Module):
    def __init__(self, feat_channel, corr_size):
        super(CorrelationLayer, self).__init__()

        self.pool_layer = nn.AdaptiveAvgPool2d((corr_size, corr_size))

        self.corr_reduce = nn.Sequential(
            nn.Conv2d(corr_size * corr_size, feat_channel, kernel_size=1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(),
            nn.Conv2d(feat_channel, feat_channel, 3, padding=1),
        )
        # self.Dnorm = nn.InstanceNorm2d(feat_channel)
        self.depth_corr_reduce = nn.Sequential(
            nn.Conv2d(corr_size * corr_size, feat_channel, kernel_size=1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(),
            nn.Conv2d(feat_channel, feat_channel, 3, padding=1),
        )

        self.rgb_feat_adapt = nn.Sequential(
            nn.Conv2d(feat_channel * 2, feat_channel, 1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(inplace=True)
        )
        self.depth_feat_adapt = nn.Sequential(
            nn.Conv2d(feat_channel * 2, feat_channel, 1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # calculate correlation map
        RGB_feat_downsize = F.normalize(self.pool_layer(x[0]))
        RGB_feat_norm = F.normalize(x[0])
        RGB_corr, _ = corr_fun(RGB_feat_downsize, RGB_feat_norm)

        # calculate correlation map
        RGB_feat_downsize = F.normalize(self.pool_layer(x[0]))
        RGB_feat_norm = F.normalize(x[0])
        RGB_corr, _ = corr_fun(RGB_feat_downsize, RGB_feat_norm)

        Depth_feat_downsize = F.normalize(self.pool_layer(x[1]))
        Depth_feat_norm = F.normalize(x[1])
        Depth_corr, _ = corr_fun(Depth_feat_downsize, Depth_feat_norm)

        corr = (RGB_corr + Depth_corr) / 2
        Red_corr = self.corr_reduce(corr)
        Depth_corr = self.depth_corr_reduce(corr)

        # beta cond
        new_feat = torch.cat([x[0], Red_corr], 1)
        new_feat = self.rgb_feat_adapt(new_feat)

        depth_feat = torch.cat([x[1], Depth_corr], 1)
        depth_feat = self.depth_feat_adapt(depth_feat)
        return new_feat, depth_feat

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels, with_pac=True):
        super(decoder, self).__init__()
        self.with_pac = with_pac
        if with_pac:
            self.pac = PacConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.norm = nn.InstanceNorm2d(in_channels)
        self.decoding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, feat, guide):
        if self.with_pac:
            feat = self.norm(self.pac(feat, guide)).relu()
        output = self.decoding(feat)
        return output

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class RAttention_Diff(nn.Module):
    '''This part of code is refactored based on https://github.com/Serge-weihao/CCNet-Pure-Pytorch. 
       We would like to thank Serge-weihao and the authors of CCNet for their clear implementation.'''
    def __init__(self, in_dim):
        super(RAttention_Diff, self).__init__()
        # print(in_dim//8)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))


    def forward(self, x_exmplar, x_query):
        m_batchsize, _, height, width = x_query.size()
        proj_query = self.query_conv(x_query)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)

        proj_query_LR = torch.diagonal(proj_query, 0, 2, 3)
        proj_query_RL = torch.diagonal(torch.transpose(proj_query, 2, 3), 0, 2, 3)

        proj_query_value = self.query_value_conv(x_query)
        proj_query_value_H = proj_query_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_query_value_W = proj_query_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_query_value_LR = torch.diagonal(proj_query_value, 0, 2, 3)
        proj_query_value_RL = torch.diagonal(torch.transpose(proj_query_value, 2, 3), 0, 2, 3)

        proj_key = self.key_conv(x_exmplar)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_key_LR = torch.diagonal(proj_key, 0, 2, 3).permute(0, 2, 1).contiguous()
        proj_key_RL = torch.diagonal(torch.transpose(proj_key, 2, 3), 0, 2, 3).permute(0, 2, 1).contiguous()

        proj_value = self.value_conv(x_exmplar)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_LR = torch.diagonal(proj_value, 0, 2, 3)
        proj_value_RL = torch.diagonal(torch.transpose(proj_value, 2, 3), 0, 2, 3)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        energy_LR = torch.bmm(proj_key_LR, proj_query_LR)
        energy_RL = torch.bmm(proj_key_RL, proj_query_RL)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        out_LR = self.softmax(torch.bmm(proj_value_LR, energy_LR).unsqueeze(-1))
        out_RL = self.softmax(torch.bmm(proj_value_RL, energy_RL).unsqueeze(-1))

        query_out_H = torch.bmm(proj_query_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1,
                                                                                 height).permute(0, 2, 3, 1)
        query_out_W = torch.bmm(proj_query_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1,
                                                                                 width).permute(0, 2, 1, 3)

        query_out_LR = self.softmax(torch.bmm(proj_query_value_LR, energy_LR).unsqueeze(-1))
        query_out_RL = self.softmax(torch.bmm(proj_query_value_RL, energy_RL).unsqueeze(-1))

        examplar_att = out_H + out_W + out_LR + out_RL
        query_att = query_out_H + query_out_W + query_out_LR + query_out_RL


        return (self.gamma_1 * examplar_att + x_exmplar,
                self.gamma_2 * query_att + x_query)


class Relation_Attention_Diff(nn.Module):
    def __init__(self, in_channels_x, in_channels_y):
        super(Relation_Attention_Diff, self).__init__()
        self.in_channels_x = in_channels_x
        self.in_channels_y = in_channels_y
        self.in_channels_xy = in_channels_x + in_channels_y

        if self.in_channels_x == 256:
            corr_size = 8
        else:
            corr_size = 32
        self.examplar_corr_layer = CorrelationLayer(feat_channel=self.in_channels_x, corr_size=corr_size)
        self.examplar_decoder = decoder(self.in_channels_x, self.in_channels_y)

        self.query_corr_layer = CorrelationLayer(feat_channel=self.in_channels_x, corr_size=corr_size)
        self.query_decoder = decoder(self.in_channels_x, self.in_channels_y)

        inter_channels_x = self.in_channels_x // 4
        self.rgb_conv_examplar = nn.Sequential(
            nn.Conv2d(self.in_channels_x, inter_channels_x, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels_x), nn.ReLU(inplace=False))
        self.rgb_conv_query = nn.Sequential(nn.Conv2d(self.in_channels_x, inter_channels_x, 3, padding=1, bias=False),
                                            nn.BatchNorm2d(inter_channels_x), nn.ReLU(inplace=False))

        self.rgb_ra = RAttention_Diff(inter_channels_x)
        self.rgb_conv_examplar_tail = nn.Sequential(
            nn.Conv2d(inter_channels_x, self.in_channels_x, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_x), nn.ReLU(inplace=False))
        self.rgb_conv_query_tail = nn.Sequential(
            nn.Conv2d(inter_channels_x, self.in_channels_x, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_x), nn.ReLU(inplace=False))

        inter_channels_y = self.in_channels_y // 4
        # print(inter_channels_y)
        self.depth_conv_examplar = nn.Sequential(
            nn.Conv2d(self.in_channels_y, inter_channels_y, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels_y), nn.ReLU(inplace=False))
        self.depth_conv_query = nn.Sequential(nn.Conv2d(self.in_channels_y, inter_channels_y, 3, padding=1, bias=False),
                                              nn.BatchNorm2d(inter_channels_y), nn.ReLU(inplace=False))

        self.depth_ra = RAttention_Diff(inter_channels_y)
        self.depth_conv_examplar_tail = nn.Sequential(
            nn.Conv2d(inter_channels_y, self.in_channels_y, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_y), nn.ReLU(inplace=False))
        self.depth_conv_query_tail = nn.Sequential(
            nn.Conv2d(inter_channels_y, self.in_channels_y, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_y), nn.ReLU(inplace=False))

        self.examplar_fusion = nn.Sequential(nn.Conv2d(self.in_channels_xy, self.in_channels_x, 3, 1, 1),
                                             nn.BatchNorm2d(self.in_channels_x), nn.ReLU())
        self.query_fusion = nn.Sequential(nn.Conv2d(self.in_channels_xy, self.in_channels_x, 3, 1, 1),
                                          nn.BatchNorm2d(self.in_channels_x), nn.ReLU())

    def forward(self, x_exmplar, x_query, y_exmplar, y_query, recurrence=2):
        lamda = 1
        # print(x_exmplar.shape)
        x_exmplar, y_exmplar = self.examplar_corr_layer((x_exmplar, y_exmplar))
        x_exmplar = self.examplar_decoder(x_exmplar, y_exmplar * lamda)

        x_query, y_query = self.query_corr_layer((x_query, y_query))
        x_query = self.query_decoder(x_query, y_query * lamda)

        x_exmplar = self.rgb_conv_examplar(x_exmplar)
        x_query = self.rgb_conv_query(x_query)
        # print(y_exmplar.shape)
        y_exmplar = self.depth_conv_examplar(y_exmplar)
        y_query = self.depth_conv_query(y_query)

        for i in range(recurrence):
            x_exmplar, x_query = self.rgb_ra(x_exmplar, x_query)
            y_exmplar, y_query = self.depth_ra(y_exmplar, y_query)

        x_exmplar = self.rgb_conv_examplar_tail(x_exmplar)
        x_query = self.rgb_conv_query_tail(x_query)

        y_exmplar = self.depth_conv_examplar_tail(y_exmplar)
        y_query = self.depth_conv_query_tail(y_query)

        exmplar_concatenation = torch.cat([x_exmplar, y_exmplar], 1)
        exmplar = self.examplar_fusion(exmplar_concatenation)
        query_concatenation = torch.cat([x_query, y_query], 1)
        query = self.query_fusion(query_concatenation)

        return exmplar, query
        

class RAttention(nn.Module):
    '''This part of code is refactored based on https://github.com/Serge-weihao/CCNet-Pure-Pytorch.
       We would like to thank Serge-weihao and the authors of CCNet for their clear implementation.'''
    def __init__(self, in_dim):
        super(RAttention, self).__init__()
        # print(in_dim//8)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))

    def forward(self, x_exmplar, x_query):
        m_batchsize, _, height, width = x_query.size()
        proj_query = self.query_conv(x_query)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)

        proj_query_LR = torch.diagonal(proj_query, 0, 2, 3)
        proj_query_RL = torch.diagonal(torch.transpose(proj_query, 2, 3), 0, 2, 3)
        # .contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_key = self.key_conv(x_exmplar)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_key_LR = torch.diagonal(proj_key, 0, 2, 3).permute(0, 2, 1).contiguous()
        proj_key_RL = torch.diagonal(torch.transpose(proj_key, 2, 3), 0, 2, 3).permute(0, 2, 1).contiguous()

        proj_value = self.value_conv(x_exmplar)
        # proj_value = self.value_conv(x_exmplar*motion)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_LR = torch.diagonal(proj_value, 0, 2, 3)
        proj_value_RL = torch.diagonal(torch.transpose(proj_value, 2, 3), 0, 2, 3)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        # energy_LR = torch.bmm(proj_query_LR, proj_key_LR)
        # energy_RL = torch.bmm(proj_query_RL, proj_key_RL)
        energy_LR = torch.bmm(proj_key_LR, proj_query_LR)
        energy_RL = torch.bmm(proj_key_RL, proj_query_RL)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        out_LR = self.softmax(torch.bmm(proj_value_LR, energy_LR).unsqueeze(-1))
        out_RL = self.softmax(torch.bmm(proj_value_RL, energy_RL).unsqueeze(-1))

        # print(out_H.size())
        # print(out_LR.size())
        # print(out_RL.size())

        # # if motion == None:
        return self.gamma_1 * (out_H + out_W + out_LR + out_RL) + x_exmplar, self.gamma_2 * (
                out_H + out_W + out_LR + out_RL) + x_query,


class Relation_Attention(nn.Module):
    def __init__(self, in_channels_x, in_channels_y):
        super(Relation_Attention, self).__init__()
        self.in_channels_x = in_channels_x
        self.in_channels_y = in_channels_y
        self.in_channels_xy = in_channels_x + in_channels_y

        if self.in_channels_x == 256:
            corr_size = 8
        else:
            corr_size = 32
        self.examplar_corr_layer = CorrelationLayer(feat_channel=self.in_channels_x, corr_size=corr_size)
        self.examplar_decoder = decoder(self.in_channels_x, self.in_channels_y)

        self.query_corr_layer = CorrelationLayer(feat_channel=self.in_channels_x, corr_size=corr_size)
        self.query_decoder = decoder(self.in_channels_x, self.in_channels_y)

        inter_channels_x = self.in_channels_x // 4
        self.rgb_conv_examplar = nn.Sequential(
            nn.Conv2d(self.in_channels_x, inter_channels_x, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels_x), nn.ReLU(inplace=False))
        self.rgb_conv_query = nn.Sequential(nn.Conv2d(self.in_channels_x, inter_channels_x, 3, padding=1, bias=False),
                                            nn.BatchNorm2d(inter_channels_x), nn.ReLU(inplace=False))

        self.rgb_ra = RAttention(inter_channels_x)
        self.rgb_conv_examplar_tail = nn.Sequential(
            nn.Conv2d(inter_channels_x, self.in_channels_x, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_x), nn.ReLU(inplace=False))
        self.rgb_conv_query_tail = nn.Sequential(
            nn.Conv2d(inter_channels_x, self.in_channels_x, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_x), nn.ReLU(inplace=False))

        inter_channels_y = self.in_channels_y // 4
        # print(inter_channels_y)
        self.depth_conv_examplar = nn.Sequential(
            nn.Conv2d(self.in_channels_y, inter_channels_y, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels_y), nn.ReLU(inplace=False))
        self.depth_conv_query = nn.Sequential(nn.Conv2d(self.in_channels_y, inter_channels_y, 3, padding=1, bias=False),
                                              nn.BatchNorm2d(inter_channels_y), nn.ReLU(inplace=False))

        self.depth_ra = RAttention(inter_channels_y)
        self.depth_conv_examplar_tail = nn.Sequential(
            nn.Conv2d(inter_channels_y, self.in_channels_y, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_y), nn.ReLU(inplace=False))
        self.depth_conv_query_tail = nn.Sequential(
            nn.Conv2d(inter_channels_y, self.in_channels_y, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels_y), nn.ReLU(inplace=False))

        self.examplar_fusion = nn.Sequential(nn.Conv2d(self.in_channels_xy, self.in_channels_x, 3, 1, 1),
                                             nn.BatchNorm2d(self.in_channels_x), nn.ReLU())
        self.query_fusion = nn.Sequential(nn.Conv2d(self.in_channels_xy, self.in_channels_x, 3, 1, 1),
                                          nn.BatchNorm2d(self.in_channels_x), nn.ReLU())

    def forward(self, x_exmplar, x_query, y_exmplar, y_query, recurrence=2):
        lamda = 1
        # print(x_exmplar.shape)
        x_exmplar, y_exmplar = self.examplar_corr_layer((x_exmplar, y_exmplar))
        x_exmplar = self.examplar_decoder(x_exmplar, y_exmplar * lamda)

        x_query, y_query = self.query_corr_layer((x_query, y_query))
        x_query = self.query_decoder(x_query, y_query * lamda)

        x_exmplar = self.rgb_conv_examplar(x_exmplar)
        x_query = self.rgb_conv_query(x_query)
        # print(y_exmplar.shape)
        y_exmplar = self.depth_conv_examplar(y_exmplar)
        y_query = self.depth_conv_query(y_query)

        for i in range(recurrence):
            x_exmplar, x_query = self.rgb_ra(x_exmplar, x_query)
            y_exmplar, y_query = self.depth_ra(y_exmplar, y_query)

        x_exmplar = self.rgb_conv_examplar_tail(x_exmplar)
        x_query = self.rgb_conv_query_tail(x_query)

        y_exmplar = self.depth_conv_examplar_tail(y_exmplar)
        y_query = self.depth_conv_query_tail(y_query)

        exmplar_concatenation = torch.cat([x_exmplar, y_exmplar], 1)
        exmplar = self.examplar_fusion(exmplar_concatenation)
        query_concatenation = torch.cat([x_query, y_query], 1)
        query = self.query_fusion(query_concatenation)

        return exmplar, query



class CoattentionModel(nn.Module):  # spatial and channel attention module
    def __init__(self, num_classes=1, all_channel=256, all_dim=26 * 26):  # 473./8=60 416./8=52
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate1 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate2 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.globalAvgPool = nn.AvgPool2d(26, stride=1)
        self.fc1 = nn.Linear(in_features=256*2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=256)
        self.fc3 = nn.Linear(in_features=256*2, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, exemplar, query):

        # spatial co-attention
        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim=1)  #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        
        # spacial attention
        input1_mask = self.gate1(torch.cat([input1_att, input2_att], dim=1))
        input2_mask = self.gate2(torch.cat([input1_att, input2_att], dim=1))
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)

        # channel attention
        out_e = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_e = out_e.view(out_e.size(0), -1)
        out_e = self.fc1(out_e)
        out_e = self.relu(out_e)
        out_e = self.fc2(out_e)
        out_e = self.sigmoid(out_e)
        out_e = out_e.view(out_e.size(0), out_e.size(1), 1, 1)
        out_q = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_q = out_q.view(out_q.size(0), -1)
        out_q = self.fc3(out_q)
        out_q = self.relu(out_q)
        out_q = self.fc4(out_q)
        out_q = self.sigmoid(out_q)
        out_q = out_q.view(out_q.size(0), out_q.size(1), 1, 1)

        # apply dual attention masks
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input2_att = out_e * input2_att
        input1_att = out_q * input1_att

        # concate original feature
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = self.bn1(input1_att)
        input2_att = self.bn2(input2_att)
        input1_att = self.prelu(input1_att)
        input2_att = self.prelu(input2_att)

        return input1_att, input2_att  # shape: NxCx

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

if __name__ == '__main__':
    model = VMD_Network().cuda()
    initialize_weights(model)
    exemplar = torch.rand(2, 3, 416, 416)
    query = torch.rand(2, 3, 416, 416)
    other = torch.rand(2, 3, 416, 416)
    exemplar_pre, query_pre, other_pre = model(exemplar, query, other)
    print(exemplar_pre.shape)
    print(query_pre.shape)
