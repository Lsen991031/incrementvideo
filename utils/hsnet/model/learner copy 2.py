import torch.nn as nn
import torch.nn.functional as F

from .base.conv4d import CenterPivotConv4d as Conv4d
import ipdb


class HPNLearner(nn.Module):
    def __init__(self, args, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 4, 16, 32
        out = 128 
        # och1, och2 = 256, 2048
        och1, och2, och3, och4 = 256, 512, 1024, 2048

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, och1, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(och1, och2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.sqz4_encoder = nn.Sequential(nn.Conv2d(out, och1, (3, 3), padding=0, stride=2, bias=True),
        #                         nn.BatchNorm2d(och1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och1, och2, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och2),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och2, och3, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och3),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och3, och4, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och4),
        #                         nn.ReLU())        
        
        # self.decoder1 = nn.Sequential(nn.Conv2d(out, och1, (3, 3), padding=0, stride=2, bias=True),
        #                         nn.BatchNorm2d(och1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och1, och2, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och2),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och2, och3, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och3),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och3, och4, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och4),
        #                         nn.ReLU())        
        och1, och2 = 512, 2048
        self.decoder1 = nn.Sequential(nn.Conv2d(out, och1, (3, 3), padding=0, stride=2, bias=True),
                                nn.BatchNorm2d(och1),
                                nn.ReLU(),
                                nn.Conv2d(och1, och2, (3, 3), padding=1, stride=2, bias=True),
                                nn.BatchNorm2d(och2),
                                nn.ReLU())        
        self.decoder1_hmdb = nn.Sequential(nn.Conv2d(och2, och2, (3, 3), padding=1, stride=2, bias=True),
                        nn.BatchNorm2d(och2),
                        nn.ReLU(),
                        nn.Conv2d(och2, och2, (3, 3), padding=1, stride=2, bias=True),
                        nn.BatchNorm2d(och2),
                        nn.ReLU()) 

        # och1, och2 = 256, 512
        # self.decoder_ucf = nn.Sequential(nn.Conv2d(out, och1, (3, 3), padding=0, stride=2, bias=True),
        #                         nn.BatchNorm2d(och1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och1, och2, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och2),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och2, och2, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och2),
        #                         nn.ReLU(),
        #                         nn.Conv2d(och2, och2, (3, 3), padding=1, stride=2, bias=True),
        #                         nn.BatchNorm2d(och2),
        #                         nn.ReLU())        
        och1, och2 = 128, 512
        self.decoder_ucf = nn.Sequential(nn.Conv2d(32, och1, (3, 3), padding=0, stride=2, bias=True),
                        nn.BatchNorm2d(och1),
                        nn.ReLU(),
                        nn.Conv2d(och1, och2, (3, 3), padding=1, stride=2, bias=True),
                        nn.BatchNorm2d(och2),
                        nn.ReLU())        

        self.decoder2_ucf = nn.Sequential(nn.Conv2d(och2, och2, (3, 3), padding=1, stride=2, bias=True),
                        nn.BatchNorm2d(och2),
                        nn.ReLU(),
                        nn.Conv2d(och2, och2, (3, 3), padding=1, stride=2, bias=True),
                        nn.BatchNorm2d(och2),
                        nn.ReLU())        

        self.scaler = args.scaler
        self.dataset = args.dataset

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))
        # for m in self.modules():
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)


    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):
        
        # ipdb.set_trace()
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        # print("hypercorr_pyramid----------shape is : {}".format(hypercorr_pyramid.size()))

        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # print("hypercorr_sqz4's shape is : {}".format(hypercorr_sqz4.shape))
        # print("hypercorr_sqz3's shape is : {}".format(hypercorr_sqz3.shape))
        # print("hypercorr_sqz2's shape is : {}".format(hypercorr_sqz2.shape))

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # print("hypercorr_mix432's shape is : {}".format(hypercorr_mix432.shape))

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # print("bsz----------shape is : {}".format(bsz))
        # print("ch----------shape is : {}".format(ch))
        #ch = int(ch * bsz / 8)
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # print("encoded----------shape is : {}".format(hypercorr_encoded.shape))
        # print("hypercorr_decoded----------shape is : {}".format(hypercorr_decoded.shape))

        # Decode the encoded 4D-tensor
        if self.dataset == 'ucf101':
            hypercorr_decoded = self.decoder_ucf(hypercorr_encoded)
            # print("encoded----------shape is : {}".format(hypercorr_encoded.shape))
            # print("hypercorr_decoded----------shape is : {}".format(hypercorr_decoded.shape))
            hypercorr_decoded1 = hypercorr_decoded.view(int(bsz/8), 8*512, 7, 7).mean(dim=-4)
            # print("hypercorr_decoded1----------shape is : {}".format(hypercorr_decoded1.shape))
            hypercorr_decoded1 = hypercorr_decoded1.view(8, 512, 7, 7)
            # print("hypercorr_decoded1----------shape is : {}".format(hypercorr_decoded1.shape))
            hypercorr_decoded2 = self.decoder2_ucf(hypercorr_decoded)
            hypercorr_decoded2 = hypercorr_decoded2.view(bsz, 512, -1).mean(dim=-1)
            hypercorr_decoded2 = hypercorr_decoded2.view(int(bsz/8), 8*512).mean(dim=-2)
            # print("decoded---------- is : {}".format(hypercorr_decoded))
            hypercorr_decoded2 = hypercorr_decoded2.view(8, 512)

            # hypercorr_decoded = hypercorr_decoded.view(bsz, 512, -1).mean(dim=-1)
            # hypercorr_decoded = hypercorr_decoded.view(int(bsz/8), 8*512).mean(dim=-2)
            # # print("decoded---------- is : {}".format(hypercorr_decoded))
            # hypercorr_decoded = hypercorr_decoded.view(8, 512)
        else:
            hypercorr_decoded = self.decoder1(hypercorr_encoded)
            # print("encoded----------shape is : {}".format(hypercorr_encoded.shape))
            # print("hypercorr_decoded----------shape is : {}".format(hypercorr_decoded.shape))
            # hypercorr_decoded = hypercorr_decoded.view(bsz, 2048, -1).mean(dim=-1)
            hypercorr_decoded1 = hypercorr_decoded.view(int(bsz/8), 8*2048, 7, 7).mean(dim=-4)
            hypercorr_decoded1 = hypercorr_decoded1.view(8, 2048, 7, 7)
            # print("decoded1---------- is : {}".format(hypercorr_decoded1.shape))
            # hypercorr_decoded = hypercorr_decoded.view(8, 2048)

            hypercorr_decoded2 = self.decoder1_hmdb(hypercorr_decoded)
            hypercorr_decoded2 = hypercorr_decoded2.view(bsz, 2048, -1).mean(dim=-1)
            hypercorr_decoded2 = hypercorr_decoded2.view(int(bsz/8), 8*2048).mean(dim=-2)
            # print("decoded---------- is : {}".format(hypercorr_decoded))
            hypercorr_decoded2 = hypercorr_decoded2.view(8, 2048)
        hypercorr_decoded1 = hypercorr_decoded1 * self.scaler
        # print("decoded1---------- is : {}".format(hypercorr_decoded1.shape))
        hypercorr_decoded2 = hypercorr_decoded2 * self.scaler
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # print("decoded----------shape is : {}".format(hypercorr_decoded))

        return hypercorr_decoded1, hypercorr_decoded2 # logit_mask
