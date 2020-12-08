import pdb
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from model.fcos import FCOSDetector
from dataset.VOC_dataset import VOCDataset
from tensorboardX import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=36, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0,1,2,3,4,5,6,7', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
train_dataset = VOCDataset(root_dir='./data/VOCdevkit/VOC2007',resize_size=[800,1333],
                           split='trainval',use_difficult=False,is_train=True,augment=transform)

model = FCOSDetector(mode="training")#.cuda()

print('==> Resuming from checkpoint..')
model.load_state_dict(torch.load('./checkpoint/model_Pretrained_29.pth'),strict=False)
model = torch.nn.DataParallel(model).cuda()
#model.requires_grad_(False)
count=0
for param in model.parameters():
    count+=1
    if count<=635:#Only train SFAM and the predictor Heads
        param.requires_grad = False

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(train_dataset)))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = 501

GLOBAL_STEPS = 1
LR_INIT = 2e-3
LR_END = 2e-5
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

start_epoch = 0

for epoch in range(start_epoch,EPOCHS):
    #model = FCOSDetector(mode="training")
    model.train()
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()
        #print("batch img is cuda: ", batch_imgs.is_cuda())

        #lr = lr_func()
        if GLOBAL_STEPS < WARMPUP_STEPS:
           lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == 20001:
           lr = LR_INIT * 0.1
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == 27001:
           lr = LR_INIT * 0.01
           for param in optimizer.param_groups:
              param['lr'] = lr
        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        
        loss = losses[-1].cuda()
        
        loss.mean().backward()
        writer.add_scalar("Loss/train", loss.mean().item(), epoch)
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)
        print( "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
             losses[2].mean(), cost_time, lr, loss.mean()))

        GLOBAL_STEPS += 1

    torch.save(model.state_dict(),"./checkpoint/model_Pretrained_{}.pth".format(epoch + 1))
   
