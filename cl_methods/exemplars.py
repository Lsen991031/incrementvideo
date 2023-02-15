import torch
import numpy as np
from collections import OrderedDict
from ops.transforms import *
from ops.models import TSN
from ops.dataset import TSNDataSet
from torch.utils.data import DataLoader
from ops.utils import *
from numpy.random import randint
import wandb
import torch.nn.functional as F
import time

from tqdm import *

import torch.distributed as dist 

def _get_indices(num_frames, num_segments):
    offsets = np.multiply(list(range(num_segments)), num_frames//num_segments) + randint(num_frames//num_segments,size=num_segments)
    return torch.tensor(offsets + 1)


def _remove_feat(features, idx, args):
    results = torch.cat((features[:idx,...],features[idx+1:,...]))
    return results


def _construct_exemplar_set(dloader, model, current_task, class_indexer, memory_size, args):
    '''DDP parameter'''
    rank = args.rank
    device = torch.device(args.gpu)


    model.eval()
    ex_dict = {}
    counter  = 0
    for i in current_task:
        ex_dict[class_indexer[i]] = {}

    with torch.no_grad():
        if rank == 0:
            tq_dloader = tqdm(enumerate(dloader), total=len(dloader),leave = True)
        else:
            tq_dloader = enumerate(dloader)
        for i, (input, target, props) in tq_dloader:
            
            # print(len(dloader))
            input = input.cuda()
            # print("target2222222222222222222 is {}".format(target))
            target = target
            # compute output
            outputs = model(input=input, only_feat=True)
            logits = outputs['preds']
            feat = outputs['feat']

            feat = feat.mean(1)

            for j in range(target.size(0)):
                k = props[0][j]
                vn = props[1][j]

                v = []
                vind = []
                v.append(feat[j])
                vind.append(props[2][j])

                if int(target[j]) in ex_dict.keys():
                    for m in range(len(v)):
                        ex_dict[int(target[j])].update({counter:(k,v[m],vn,vind[m])})
                        counter +=1

    exemplar_list = []

    for i in current_task:
        temp_dict = ex_dict[class_indexer[i]]
        paths = []
        features = []
        nframes = []
        inds = []

        for k, v in enumerate(temp_dict.items()):
            f_path = v[1][0]
            feat = v[1][1]
            feat = feat / torch.norm(feat,p=2)
            nframe = v[1][2]
            frame_ind = torch.unique(v[1][3],sorted=True)
            paths.append(f_path)
            features.append(feat)
            nframes.append(nframe)
            inds.append(frame_ind)

        features = torch.stack(features)
        class_mean = torch.mean(features,axis=0)
        class_mean = class_mean / torch.norm(class_mean,p=2)

        exemplar_i = {}

        step_t = 0
        mu = class_mean
        w_t = mu

        while True:
            if len(exemplar_i.keys())>=memory_size:
                break
            if features.size(0) == 0:
                break

            tmp_t = torch.matmul(features,mu) # dot w_t, features
            index = torch.argmax(tmp_t)
            w_t = w_t + mu - features[index]
            step_t += 1

            if paths[index] not in exemplar_i.keys():
                if args.store_frames=='entire':
                    exemplar_i[paths[index]] = (step_t,nframes[index],inds[index])
                else:
                    exemplar_i[paths[index]] = (step_t,len(inds[index]),inds[index])

            features = _remove_feat(features,index,args)
            del paths[index]
            del nframes[index]
            del inds[index]

        exemplar_i = OrderedDict(sorted(exemplar_i.items(), key=lambda x: x[1][0]))
        # print(exemplar_i)
        exemplar_list.append(exemplar_i)

    return exemplar_list

def _reduce_exemplar_set(ex_list, memory_size):
    reduced_list = []

    for i in range(len(ex_list)):
        ex_i = list(ex_list[i].items())
        reduced_list.append(OrderedDict(ex_i[:memory_size]))

    return reduced_list

def compute_mask(args, model, exemplar_loader):
    with model.no_sync():
        '''DDP parameter'''
        rank = args.rank
        device = torch.device(args.gpu)

        model = model.eval()
        # freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

        if rank == 0:
            tq_exemplar_loader = tqdm(enumerate(exemplar_loader), total=len(exemplar_loader),leave = True)
        else:
            tq_exemplar_loader = enumerate(exemplar_loader)
        for i_num, (input, target, props) in tq_exemplar_loader:
            # if rank == 0:
            #     print("target11111111111111111111111111111111 is {}".format(target))
                # print(len(exemplar_loader))
            input_video = input.cuda() # 4, 24, 224, 224
            print("input_video.shape is {}".format(input_video.shape)) 
            # print(input.shape)
            a = input_video.shape
            # print(props.shape)

            # temp_input = input_video.view(a[0], 8, 3, 224, 224)
            # temp_input = temp_input.mean(-4)
            # inputs_f = temp_input.view(a[0], 3, 224 , 224) # 4, 3, 224, 224
            # inputs_f = inputs_f.repeat(1,8,1,1)
            # b = inputs_f.shape
            # print("a is {}".format(a))
            # print("b is {}".format(b))
            # learnable_para = torch.nn.Parameter(torch.ones(b), requires_grad=True).cuda()


            weight = (torch.ones(a[0], 8)/a[0]).requires_grad_(True)  # bs, T
            # prompt = torch.zeros_like(inputs_video[:,:,0,:,:]).unsqueeze(2).requires_grad_(True)      # bs, c, 1, H, W

            learnable_para = torch.ones([a[0], 3, 224, 224]) - 1
            learnable_para = learnable_para.cuda()
            learnable_para.requires_grad = True
            optimizer = torch.optim.Adam([{'params': [weight], 'lr': 0.1}, {'params': [learnable_para], 'lr': 0.01}], amsgrad=True)

            lr = 0.1
            inputs_image = []

            time.sleep(0.01)
            if rank == 0:
                tq_range = tqdm(range(1000))
            else:
                tq_range = range(1000)
            for i in tq_range:
            # for i in range(10000):
                # optimizer = torch.optim.Adam([{'params': [weight], 'lr': 0.001}, {'params': [learnable_para], 'lr': lr}], amsgrad=True)
                temp_input = input_video.view(a[0], 8, 3, 224, 224)
                # print("temp_input.shape is {}".format(temp_input.shape)) 
                inputs_f = temp_input.cuda() * (F.softmax(weight,dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).cuda()
                # print("inputs_f.shape is {}".format(inputs_f.shape)) 
                inputs_f = torch.sum(inputs_f, dim=1)   #bs, c, H, W
                # print("inputs_f.shape is {}".format(inputs_f.shape)) 
                inputs_image = inputs_f + learnable_para
                # print("inputs_image.shape is {}".format(inputs_image.shape)) 
                inputs_f = inputs_f.repeat(1,8,1,1)
                inputs_image = inputs_image.repeat(1,8,1,1)
                b = inputs_f.shape
                # print("inputs_image.shape is {}".format(inputs_image.shape)) 
                # print("inputs_image is {}".format(inputs_image)) 

                
                 # .repeat(1,8,1,1) # 
                # print(temp_input.shape)
                target = target.cuda()

                inputs_all = torch.cat([inputs_f,inputs_image],dim=0)
                # print(inputs_all.shape) # torch.Size([8, 8, 2048])
                outputs_all = model(input=inputs_all)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                # print(outputs_all['feat'].shape)
                logit_f = outputs_all['feat'][:a[0]] # torch.Size([4, 8, 512])
                # print("logit_f.shape is {}".format(logit_f.shape))
                logit_fp = outputs_all['feat'][a[0]:] # torch.Size([4, 8, 512])
                # print("logit_fp.shape is {}".format(logit_fp.shape))
                preds_f = outputs_all['preds'][:a[0]] # torch.Size([4, 51])
                # print("preds_f.shape is {}".format(preds_f.shape))
                preds_fp = outputs_all['preds'][a[0]:] # torch.Size([4, 51])
                # print("preds_fp.shape is {}".format(preds_fp.shape))

                # print(temp_preds.shape)
                with torch.no_grad():
                    outputs = model(input=input_video)
                # print(outputs)
                logit = outputs['feat'] # torch.Size([4, 8, 512])
                preds = outputs['preds'] # torch.Size([4, 51])
                # print(preds.shape)

                # print(target.shape) # torch.Size([4])

                confidence1 = torch.nn.CrossEntropyLoss(reduction="mean")(preds_f, target)
                confidence2 = torch.nn.CrossEntropyLoss(reduction="mean")(preds_fp, target)
                dist1 = torch.nn.MSELoss(reduction="mean")(logit, logit_f)
                dist2 = torch.nn.MSELoss(reduction="mean")(logit, logit_fp)

                loss = dist1 + dist2 + 10 * confidence1 + 10 * confidence2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if rank == 0:
                    # if i % 100 == 0:
                    #     print("learnable_para is {}".format(learnable_para))
                    #     print("weight is {}".format(weight))
                    #     print("loss is {}".format(loss))
                    #     print("confidence1 is {}".format(confidence1))
                    #     print("confidence2 is {}".format(confidence2))
                    #     print("dist1 is {}".format(dist1))
                    #     print("dist2 is {}".format(dist2))
                    #     print("target is {}".format(target))
                    #     print("preds_f is {}".format(preds_f))
                    #     print("preds_fp is {}".format(preds_fp))
                    wandb.log({"task_a/loss1": loss,
                        "task_a/dist1": dist1,
                        "task_a/dist2": dist2,
                        "task_a/confidence1": confidence1,
                        "task_a/confidence2": confidence2,
                        })
                if i % 5000 == 0:
                    lr = lr / 10.0

            # save frame, prompt and weights    
            # weight = weight.detach().cpu()
            # learnable_para = learnable_para.cpu() # 4, 3, 224, 224
            # print(inputs_image[0].shape)
            # inputs_image = inputs_image.view(a[0], 8, 3, 224, 224)
            # inputs_image = inputs_image.mean(-4)
            # inputs_image = inputs_image.view(a[0], 3, 224, 224)

            if not os.path.exists(args.exemplar_path) and rank == 1:
                # print('creating folder ' + args.exemplar_path)
                os.mkdir(args.exemplar_path)

            for j in range(a[0]):
                filename = os.path.join(args.exemplar_path, props[0][j] , 'img.pth')
                # print(filename)
                torch.save(inputs_image[j].data, filename)


            # del outputs, outputs_all, feat, temp_feat, preds, temp_preds, inputs_image, learnable_para
            torch.cuda.empty_cache()
            
            # break

            # if rank == 0:
                # save_condense(args, condense) 
            dist.barrier()

def manage_exemplar_set(args, age, current_task, current_head, class_indexer, prefix):

    '''DDP parameter'''
    rank = args.rank
    device = torch.device(args.gpu)

    model = TSN(args, num_class=current_head,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                age=age,cur_task_size=len(current_task),training=True,fine_tune=True)

    print("Construct Exemplar Set")
    if args.budget_type == 'fixed':
        exemplar_per_class = args.K//current_head
    else:
        exemplar_per_class = args.K

    if age > 0:
        exemplar_dict = load_exemplars(args, device)
        exemplar_list = exemplar_dict[age-1]
    else:
        exemplar_dict = {}
        exemplar_list = None

    scale_size = model.scale_size
    input_size = model.input_size

    normalize = GroupNormalize(model.input_mean, model.input_std)

    print("Load the Model")
    ckpt_path = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp), 'task_{:03d}.pth.tar'.format(age))
    sd = torch.load(ckpt_path, map_location=device)# torch.device('cpu'))
    print("exemplars.device is : {}".format(device))
    sd = sd['state_dict']
    state_dict = dict()
    for k, v in sd.items():
        state_dict[k[7:]] = v
    model.load_state_dict(state_dict)

    print(model.new_fc)

    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    '''Wrap model with DDP'''        
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                        find_unused_parameters=True,
                                                        broadcast_buffers=False,) 
    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    print("Exemplar per class : {}".format(exemplar_per_class))
    # print("class_indexer per class : {}".format(class_indexer))

    # Construct Exemplar Set for the Current Task
    transform_ex = torchvision.transforms.Compose([
                                            GroupScale(scale_size),
                                            GroupCenterCrop(input_size),
                                            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                            normalize,
                                            ])

    train_dataset_for_exemplar = TSNDataSet(args.root_path, args.train_list, current_task, class_indexer,
                            num_segments=args.num_segments, random_shift=False, new_length=1,
                            modality='RGB',image_tmpl=prefix, transform=transform_ex,
                            dense_sample=args.dense_sample,
                            store_frames=args.store_frames)

    # exemplar_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_for_exemplar, shuffle=False)

    train_loader_for_exemplar = DataLoader(train_dataset_for_exemplar, batch_size=args.exemplar_batch_size,
                        # sampler=exemplar_sampler, 
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False)

    current_task_exemplar = _construct_exemplar_set(train_loader_for_exemplar,model,current_task,class_indexer,exemplar_per_class,args)

    if rank == 0:
        if not os.path.exists(args.exemplar_path):
            print('creating folder ' + args.exemplar_path)
            os.mkdir(args.exemplar_path)
        for ex_i in current_task_exemplar:
            # print(len(current_task_exemplar))
            # print(current_task_exemplar)
            # print(list(ex_i.keys()))
            for key_file in list(ex_i.keys()):
                a = key_file.split('/')
                if not os.path.exists(os.path.join(args.exemplar_path, a[0])):
                    os.mkdir(os.path.join(args.exemplar_path, a[0]))
                if not os.path.exists(os.path.join(args.exemplar_path, key_file)):    
                    os.mkdir(os.path.join(args.exemplar_path, key_file))
    dist.barrier()

    # Construct Exemplar Set for the Current Task
    transform_ex = torchvision.transforms.Compose([
                                            GroupScale(scale_size),
                                            GroupCenterCrop(input_size),
                                            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                            normalize,
                                            ])

    cur_exemplar_dataset = TSNDataSet(args.root_path, args.train_list, current_task, class_indexer,
                                num_segments=args.num_segments, random_shift=False, new_length=1,
                                modality='RGB',image_tmpl = prefix, transform=transform_ex,
                                dense_sample=args.dense_sample, exemplar_list=current_task_exemplar,
                                exemplar_only=True, is_entire=(args.store_frames=='entire'), is_img_end=False)

    # print("current_task_exemplar is {}".format(current_task_exemplar))
    exemplar_sampler = torch.utils.data.distributed.DistributedSampler(cur_exemplar_dataset, shuffle=False)

    train_loader_for_cur_exemplar = DataLoader(cur_exemplar_dataset, batch_size=args.exemplar_batch_size,
                        sampler=exemplar_sampler, 
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False)
    
    ppp = compute_mask(args, model, train_loader_for_cur_exemplar)

    if age > 0:
        # Reduce Exemplar Set
        if args.budget_type == 'fixed':
            exemplar_list = _reduce_exemplar_set(exemplar_list, exemplar_per_class)
        exemplar_list = exemplar_list + current_task_exemplar
    else:
        exemplar_list = current_task_exemplar

    del model # ?????????????????????????
    torch.cuda.empty_cache() # ?????????????????????????

    exemplar_dict[age] = exemplar_list
    print(exemplar_list)
    if rank == 0:
        save_exemplars(args, exemplar_dict)
    dist.barrier()

