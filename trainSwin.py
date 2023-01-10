# threeD_train_final.py: Define Training Loop
import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from SwinLarge import SpatialSoftmax3D, SwinUnet
from threeD_dataLoader import sample_data_diffTask_2
import pickle
import torch
import cv2
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
from threeD_viz_video import generateVideo
from threeD_viz_image import generateImage
from functools import partial
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_dir', type=str, default='./', help='Experiment path') #Change
parser.add_argument('--exp', type=str, default='singlePeople_SwinTrans_0301', help='Name of experiment')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') 
parser.add_argument('--batch_size', type=int, default=32, help='Batch size,32')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--window', type=int, default=10, help='window around the time step')
parser.add_argument('--subsample', type=int, default=1, help='subsample tile res')
parser.add_argument('--linkLoss', type=bool, default=True, help='use link loss') # Find min and max link
parser.add_argument('--epoch', type=int, default=100, help='The time steps you want to subsample the dataset to,500')
parser.add_argument('--numwork', type=int, default=12, help='The number of workers')
parser.add_argument('--ckpt', type=str, default ='singlePerson_0.0001_10_best', help='loaded ckpt file') # Enter link of trained model
parser.add_argument('--eval', type=bool, default=False, help='Set true if eval time') # Evaluation with test data. 2 Mode: Loading trained model and evaluate with test set, Training and Evaluation with evaluation set. 
parser.add_argument('--test_dir', type=str, default ='./', help='test data path') # Link to test data
parser.add_argument('--exp_image', type=bool, default=True, help='Set true if export predictions as images')
parser.add_argument('--exp_video', type=bool, default=True, help='Set true if export predictions as video')
parser.add_argument('--exp_data', type=bool, default=False, help='Set true if export predictions as raw data')
parser.add_argument('--exp_L2', type=bool, default=True, help='Set true if export L2 distance')
parser.add_argument('--train_continue', type=bool, default=False, help='Set true if eval time')
args = parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) #Khởi tạo weight
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Vị trí thực khi chưa chuẩn hóa
def get_spatial_keypoint(keypoint):
    b = np.reshape(np.array([-100, -100, -1800]), (1,1,3))
    resolution = 100
    max = 19
    spatial_keypoint = keypoint * max * resolution + b
    return spatial_keypoint 

# Ước tính khoảng cách thực giữa dự đoán và kết quả
def get_keypoint_spatial_dis(keypoint_GT, keypoint_pred):
    dis = get_spatial_keypoint(keypoint_pred) - get_spatial_keypoint(keypoint_GT)
    # mean = np.reshape(np.mean(dis, axis=0), (21,3))
    return dis 

# Loại bỏ small value
def remove_small(heatmap, threshold, rank):
    z = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4]).cuda(rank)
    heatmap = torch.where(heatmap<threshold, z, heatmap)
    return heatmap 

# Link loss
def check_link(min, max, keypoint, rank):

    # print (torch.max(max), torch.min(min))

    BODY_25_pairs = np.array([
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12],
    [12, 13], [13, 14], [1, 0], [14, 15], [15, 16], [14, 17], [11, 18], [18, 19], [11, 20]])

    # o = torch.ones(keypoint.shape[0], keypoint.shape[1], keypoint.shape[2]).cuda(rank)
    # keypoint = torch.where(torch.isnan(keypoint), o, keypoint)

    keypoint_output = torch.ones(keypoint.shape[0],20).cuda(rank)

    for f in range(keypoint.shape[0]):
        for i in range(20):

            a = keypoint[f, BODY_25_pairs[i, 0]]
            b = keypoint[f, BODY_25_pairs[i, 1]]
            s = torch.sum((a - b)**2)

            if s < min[i]:
                keypoint_output[f,i] = min[i] -s
            elif s > max[i]:
                keypoint_output[f,i] = s - max[i]
            else:
                keypoint_output[f,i] = 0

    return keypoint_output #Loss cho độ dài các khớp luôn nằm trong khoảng cho phép

if not os.path.exists(args.exp_dir + 'ckpts'):
    os.makedirs(args.exp_dir + 'ckpts')

# Tạo folder log
if not os.path.exists(args.exp_dir + 'log'):
    os.makedirs(args.exp_dir + 'log')

if not os.path.exists(args.exp_dir + 'predictions'):
    os.makedirs(args.exp_dir + 'predictions')
    os.makedirs(args.exp_dir + 'predictions/image')
    os.makedirs(args.exp_dir + 'predictions/video')
    os.makedirs(args.exp_dir + 'predictions/L2')
    os.makedirs(args.exp_dir + 'predictions/data')

# use_gpu = torch.cuda.is_available()
# device = 'cuda:0' if use_gpu else 'cpu'
# use_gpu = True
# device = 'cuda:1'
num_gpus = torch.cuda.device_count()


# Chuẩn bị data for training và validation
# args.exp_dir  -> /tactile_keypoint_data/
#               -> /singlePerson_test/

print (f"Name of experiment: {args.exp}, Window size: {args.window}, Subsample: {args.subsample}")

def run_training_process_on_given_gpu(rank, num_gpus):
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', rank=rank,
                    world_size=num_gpus, init_method='env://')
    
    model = SwinUnet()
    softmax = SpatialSoftmax3D(20, 20, 18, 21) # trả về heatmap và ước tính keypoint từ heatmap predicted

    model = model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    
    softmax.cuda(rank)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (f"Total parameters: {pytorch_total_params}")
    criterion = nn.MSELoss()
    criterion.cuda(rank)
    
    data_path = "/LOCAL2/anguyen/faic/lthieu/6DOFTactile/train/batch_data/"
    mask = []
    train_dataset = sample_data_diffTask_2(data_path, args.window, args.subsample, "train")
    train_sample = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False, num_workers=4*num_gpus, pin_memory=True, sampler=train_sample)
    
    print ("Training set size:", len(train_dataset))

    val_dataset = sample_data_diffTask_2(data_path, args.window, args.subsample, "val")
    val_sample = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False, num_workers=4*num_gpus, pin_memory=True, sampler=val_sample)
    print ("Validation set size: ", len(val_dataset))
    
    test_dataset = sample_data_diffTask_2(data_path, args.window, args.subsample, "test")
    test_sample = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False, num_workers=4*num_gpus, pin_memory=True, sampler=test_sample)
    print ("Test set size: ", len(test_dataset))
    
    if args.linkLoss:
        link_max = [0.11275216, 0.02857364, 0.03353087, 0.05807897, 0.04182064, 0.0540275, 0.04558805, 0.04482517, 0.10364685, 0.08350807, 0.0324904, 0.10430953, 0.08306233, 0.03899737, 0.04866854, 0.03326589, 0.02623637, 0.04040782, 0.02288897, 0.02690871] 
        link_min = np.zeros(20,)

        link_min = torch.tensor(link_min, dtype=torch.float).cuda(rank)
        link_max = torch.tensor(link_max, dtype=torch.float).cuda(rank)

    # Fine tune
    
    # if args.train_continue:
    #     checkpoint = torch.load( args.exp_dir + 'ckpts/' + args.ckpt + '.path.tar')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    #     epochs = checkpoint['epoch']
    #     loss = checkpoint['loss']
    #     print("ckpt loaded", loss)
    #     print("Now continue training")

    # Nếu chỉ đánh giá với tập test và pretrained model
#     if args.eval:
#         checkpoint = torch.load( args.exp_dir + 'ckpts/' + args.ckpt + '.path.tar')
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epochs = checkpoint['epoch']
#         loss = checkpoint['loss']
#         print (loss)
#         print("ckpt loaded:", args.ckpt)
#         print("Now running on val set")
# Run with test set  
    # Quay lại nếu không phải là testing thì bắt đầu train
    train_loss_list = np.zeros((1))
    val_loss_list = np.zeros((1))
    best_keypoint_loss = np.inf
    best_val_loss = np.inf

    # if args.train_continue:
    #     best_val_loss = 0.0734

    for epoch in range(args.epoch):
        print(f">>>Epoch {epoch}<<<") 
        train_loss = []
        val_loss = []
        print ('Begin training')

        bar = ProgressBar(max_value=len(train_dataloader))

        for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
            model.train(True)
            tactile = torch.tensor(sample_batched[0], dtype=torch.float).cuda(rank)
            heatmap = torch.tensor(sample_batched[1], dtype=torch.float).cuda(rank)
            keypoint = torch.tensor(sample_batched[2], dtype=torch.float).cuda(rank)
            idx = torch.tensor(sample_batched[3], dtype=torch.float).cuda(rank)
            with torch.set_grad_enabled(True):
                heatmap_out = model(tactile)
                heatmap_out = heatmap_out.reshape(-1, 21, 20, 20, 18)
                heatmap_transform = remove_small(heatmap_out.transpose(2,3), 1e-2, rank)
                keypoint_out, heatmap_out2 = softmax(heatmap_transform * 10) 

            loss_heatmap = torch.mean((heatmap_transform - heatmap)**2 * (heatmap + 0.5) * 2) * 1000
            loss_keypoint = criterion(keypoint_out, keypoint) # For metric evaluation

            if args.linkLoss:
                loss_link = torch.mean(check_link(link_min, link_max, keypoint_out, rank)) * 10
                loss = loss_heatmap + loss_link
            else:
                loss = loss_heatmap
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data.item())
            
            if i_batch % 1124 ==0 and i_batch!=0: # Cứ 50 batch lại evaluate 1 lần

                print("[%d/%d] LR: %.6f, Loss: %.6f, Heatmap_loss: %.6f, Keypoint_loss: %.6f, "
                      "k_max_gt: %.6f, k_max_pred: %.6f, k_min_gt: %.6f, k_min_pred: %.6f, "
                      "h_max_gt: %.6f, h_max_pred: %.6f, h_min_gt: %.6f, h_min_pred: %.6f" % (
                    i_batch, len(train_dataloader), get_lr(optimizer), loss.item(), loss_heatmap, loss_keypoint,
                    np.amax(keypoint.cpu().data.numpy()), np.amax(keypoint_out.cpu().data.numpy()),
                    np.amin(keypoint.cpu().data.numpy()), np.amin(keypoint_out.cpu().data.numpy()),
                    np.amax(heatmap.cpu().data.numpy()), np.amax(heatmap_out.cpu().data.numpy()),
                    np.amin(heatmap.cpu().data.numpy()), np.amin(heatmap_out.cpu().data.numpy())))


                if args.linkLoss:
                    print ("loss_heatmap:", loss_heatmap.cpu().data.numpy(),
                           "loss_link:", loss_link.cpu().data.numpy(),
                           "loss_keypoint:", loss_keypoint.cpu().data.numpy())

                print("Now running on val set")
                model.train(False)

                keypoint_l2 = []

                bar = ProgressBar(max_value=len(val_dataloader))
                for i_batch, sample_batched in bar(enumerate(val_dataloader, 0)):

                    tactile = torch.tensor(sample_batched[0], dtype=torch.float).cuda(rank)
                    heatmap = torch.tensor(sample_batched[1], dtype=torch.float).cuda(rank)
                    keypoint = torch.tensor(sample_batched[2], dtype=torch.float).cuda(rank)

                    with torch.set_grad_enabled(False):
                        heatmap_out = model(tactile)
                        heatmap_out = heatmap_out.reshape(-1, 21, 20, 20, 18)
                        heatmap_transform = remove_small(heatmap_out.transpose(2,3), 1e-2, rank)
                        keypoint_out, heatmap_out2 = softmax(heatmap_transform * 10)

                    loss_heatmap = torch.mean((heatmap_transform - heatmap)**2 * (heatmap + 0.5) * 2) * 1000
                    loss_keypoint = criterion(keypoint_out, keypoint)

                    if args.linkLoss:
                        loss_link = torch.mean(check_link(link_min, link_max, keypoint_out, rank)) * 10
                        loss = loss_heatmap + loss_link
                    else:
                        loss = loss_heatmap

#                     if i_batch % 50 == 0 and i_batch != 0:
#                         #
#                         print("[%d/%d] LR: %.6f, Loss: %.6f, Heatmap_loss: %.6f, Keypoint_loss: %.6f, "
#                           "k_max_gt: %.6f, k_max_pred: %.6f, k_min_gt: %.6f, k_min_pred: %.6f, "
#                           "h_max_gt: %.6f, h_max_pred: %.6f, h_min_gt: %.6f, h_min_pred: %.6f" % (
#                         i_batch, len(val_dataloader), get_lr(optimizer), loss.item(), loss_heatmap, loss_keypoint,
#                         np.amax(keypoint.cpu().data.numpy()), np.amax(keypoint_out.cpu().data.numpy()),
#                         np.amin(keypoint.cpu().data.numpy()), np.amin(keypoint_out.cpu().data.numpy()),
#                         np.amax(heatmap.cpu().data.numpy()), np.amax(heatmap_out.cpu().data.numpy()),
#                         np.amin(heatmap.cpu().data.numpy()), np.amin(heatmap_out.cpu().data.numpy())))
#                         #
#                         if args.linkLoss:
#                             print ("loss_heatmap:", loss_heatmap.cpu().data.numpy(),
#                                    "loss_link:", loss_link.cpu().data.numpy(),
#                                    "loss_keypoint:", loss_keypoint.cpu().data.numpy())


                    val_loss.append(loss.data.item())


                scheduler.step(np.mean(val_loss))
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,},
         args.exp_dir + 'ckpts/' + args.exp + '_' + str(args.lr)
         + '_' + str(args.window) + '_' + 'cp'+ str(epoch) + '.path.tar')
        print("Saving to ", args.exp_dir + 'ckpts/' + args.exp + '_' + str(args.lr)
         + '_' + str(args.window) + '_' + 'cp'+ str(epoch) + '.path.tar')
        if avg_val_loss < best_val_loss:
            print ("new_best_keypoint_l2:", avg_val_loss)
            best_val_loss = avg_val_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,},
               args.exp_dir + 'ckpts/' + args.exp + '_' + str(args.lr)
                + '_' + str(args.window) + '_best' + '.path.tar')
            print("Saving to ", args.exp_dir + 'ckpts/' + args.exp + '_' + str(args.lr)
                + '_' + str(args.window) + '_best' + '.path.tar')

        avg_train_loss_save = np.array([avg_train_loss])
        avg_val_loss_save = np.array([avg_val_loss])

        train_loss_list = np.append(train_loss_list,avg_train_loss_save, axis =0)
        val_loss_list = np.append(val_loss_list,avg_val_loss_save, axis = 0)

        to_save = [train_loss_list[1:],val_loss_list[1:]]
        pickle.dump(to_save, open( args.exp_dir + 'log/' + args.exp +
                                   '_' + str(args.lr) + '_' + str(args.window) + '.p', "wb" ))
        print("Save losses at: "+ args.exp_dir + 'log/' + args.exp + '_' + str(args.lr) + '_' + str(args.window) + '.p')

        print("Train Loss: %.6f, Valid Loss: %.6f" % (avg_train_loss, avg_val_loss))
        
    model.eval()
    avg_val_loss = []
    avg_val_keypoint_l2_loss = []

    tactile_GT = np.empty((1,96,96))
    heatmap_GT = np.empty((1,21,20,20,18))
    heatmap_pred = np.empty((1,21,20,20,18))
    keypoint_GT = np.empty((1,21,3))
    keypoint_pred = np.empty((1,21,3))
    tactile_GT_v = np.empty((1,96,96))
    heatmap_GT_v = np.empty((1,21,20,20,18))
    heatmap_pred_v = np.empty((1,21,20,20,18))
    keypoint_GT_v = np.empty((1,21,3))
    keypoint_pred_v = np.empty((1,21,3))
    keypoint_GT_log = np.empty((1,21,3))
    keypoint_pred_log = np.empty((1,21,3))

    bar = ProgressBar(max_value=len(test_dataloader)) # Thanh tiến trình

    c = 0
    for i_batch, sample_batched in bar(enumerate(test_dataloader, 0)):
        tactile = torch.tensor(sample_batched[0], dtype=torch.float).cuda(rank)
        heatmap = torch.tensor(sample_batched[1], dtype=torch.float).cuda(rank)
        keypoint = torch.tensor(sample_batched[2], dtype=torch.float).cuda(rank)
        tactile_frame = torch.tensor(sample_batched[3], dtype=torch.float).cuda(rank)


        with torch.set_grad_enabled(False):
            heatmap_out = model(tactile)
            heatmap_out = heatmap_out.reshape(-1, 21, 20, 20, 18) # Output shape từ model
            heatmap_transform = remove_small(heatmap_out.transpose(2,3), 1e-2, rank)
            keypoint_out, heatmap_out2 = softmax(heatmap_transform) 

        loss_heatmap = torch.mean((heatmap_transform - heatmap)**2 * (heatmap + 0.5) * 2) * 1000 # Loss heatmap
        heatmap_out = heatmap_transform

        if i_batch % 100 == 0 and i_batch != 0:
            print (i_batch, loss_heatmap)
            # loss = loss_heatmap
            # print (loss)

        '''export image'''
        # Nếu có in ra hình ảnh kết quả để kiểm nghiệm
        if args.exp_image:
            base = 0
            imageData = [heatmap.cpu().data.numpy().reshape(-1,21,20,20,18),
                             heatmap_out.cpu().data.numpy().reshape(-1,21,20,20,18),
                             keypoint.cpu().data.numpy().reshape(-1,21,3),
                             keypoint_out.cpu().data.numpy().reshape(-1,21,3),
                             tactile_frame.cpu().data.numpy().reshape(-1,96,96)]

            generateImage(imageData, args.exp_dir + 'predictions/image/', i_batch, base)

        '''log data for L2 distance and video'''
        # Lưu lại chồng các frame để in ra video
        if args.exp_video:
            if i_batch>50 and i_batch<60: #set range
                heatmap_GT_v = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
                heatmap_pred_v = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
                keypoint_GT_v = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1,21,3),axis=0)
                keypoint_pred_v = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1,21,3),axis=0)
                tactile_GT_v = np.append(tactile_GT,tactile_frame.cpu().data.numpy().reshape(-1,96,96),axis=0)

        if args.exp_L2:
            keypoint_GT_log = np.append(keypoint_GT_log, keypoint.cpu().data.numpy().reshape(-1,21,3),axis=0)
            keypoint_pred_log = np.append(keypoint_pred_log, keypoint_out.cpu().data.numpy().reshape(-1,21,3),axis=0)

        '''save data'''
        # Nếu có lưu lại kết quả dự đoán để kiểm nghiệm (append)
        if args.exp_data:
            heatmap_GT = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
            heatmap_pred = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
            keypoint_GT = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1,21,3),axis=0)
            keypoint_pred = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1,21,3),axis=0)
            tactile_GT = np.append(tactile_GT,tactile_frame.cpu().data.numpy().reshape(-1,96,96),axis=0)

            if i_batch % 20 == 0 and i_batch != 0: #set the limit to avoid overflow
                c += 1
                toSave = [heatmap_GT[1:,:,:,:,:], heatmap_pred[1:,:,:,:,:],
                              keypoint_GT[1:,:,:], keypoint_pred[1:,:,:],
                              tactile_GT[1:,:,:]]
                pickle.dump(toSave, open(args.exp_dir + 'predictions/data/' + args.exp + str(c) + '.p', "wb"))
                tactile_GT = np.empty((1,96,96))
                heatmap_GT = np.empty((1,21,20,20,18))
                heatmap_pred = np.empty((1,21,20,20,18))
                keypoint_GT = np.empty((1,21,3))
                keypoint_pred = np.empty((1,21,3))

        avg_val_loss.append(loss.data.item())
    print ("Loss:", np.mean(avg_val_loss))
    
    # Nếu có lưu lại kết quả distance giữa các keypoint để kiểm nghiệm (sau khi đã xếp chồng)
    if args.exp_L2:
        dis = get_keypoint_spatial_dis(keypoint_GT_log[1:,:,:], keypoint_pred_log[1:,:,:])
        pickle.dump(dis, open(args.exp_dir + 'predictions/L2/'+ args.exp + '_dis.p', "wb"))
        print ("keypoint_dis_saved:", dis, dis.shape)
 
    # Tạo video
    if args.exp_video:
        
        to_save = [heatmap_GT_v[1:,:,:,:,:], heatmap_pred_v[1:,:,:,:,:],
                      keypoint_GT_v[1:,:,:], keypoint_pred_v[1:,:,:],
                      tactile_GT_v[1:,:,:]]

        print (to_save[0].shape, to_save[1].shape, to_save[2].shape, to_save[3].shape, to_save[4].shape)

        generateVideo(to_save,
                  args.exp_dir + 'predictions/video/' + args.ckpt,
                  heatmap=True)
    
if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    num_gpus = torch.cuda.device_count()
    print('num_gpus: ', num_gpus)
    torch.multiprocessing.spawn(run_training_process_on_given_gpu, args=(num_gpus, ), nprocs=num_gpus, join=True)
