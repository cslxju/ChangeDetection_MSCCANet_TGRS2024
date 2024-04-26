import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics,exp_lr_scheduler_with_warmup,get_test_loaders)
import os
import logging
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix
"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
_, metadata_val = get_parser_with_args()
_, metadata_test = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""
gpu_num = input("GPU_ID:")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)

def metrics_calculation(opt, model, batch_iter, tbar, epoch, state):     
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    info = state + "_epoch {} info "
    with torch.no_grad():
        for batch_img1, batch_img2, labels in tbar:
            tbar.set_description(info.format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            cd_preds = model(batch_img1, batch_img2)
            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)
            tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                            cd_preds.data.cpu().numpy().flatten(),labels=[0,1]).ravel()              
            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    IoU = tp / (tp + fp + fn) 
    Acc = (tp + tn) / (tp + fp + tn + fn) 
    return P, R, F1, IoU, Acc
train_loader, val_loader = get_loaders(opt)
test_loader = get_test_loaders(opt)
"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')

net_name = "MSCCANet"
if not os.path.exists('./tmp'):
    os.mkdir('./tmp')
if not os.path.exists('./tmp/train'):
    os.mkdir('./tmp/train')
if not os.path.exists('./tmp/val'):
    os.mkdir('./tmp/val')
if not os.path.exists('./tmp/test'):
    os.mkdir('./tmp/test')
if not os.path.exists('./chart'):
    os.mkdir('./chart')
if not os.path.exists('./chart/test'):
    os.mkdir('./chart/test')
if not os.path.exists('./chart/train'):
    os.mkdir('./chart/train')
if not os.path.exists('./chart/val'):
    os.mkdir('./chart/val')
  
path_in = './tmp/train'
files = os.listdir(path_in)
b = []
for f in files:
    if(f[-1]=='t'):
        f = f[17:-3]
        f = int(f)
        b.append(f)
b.sort()
if(len(b)==0):   
    model = load_model(net_name,opt, dev)
    
    start_epoch = -1
    print('无保存模型，将从头开始训练！')
else:
    path_model = path_in+'/checkpoint_epoch_'+str(b[-1])+'.pt' 
    model = torch.load(path_model)
    start_epoch = b[-1]
    print('加载 epoch {} 成功！'.format(start_epoch))

criterion = get_criterion(opt)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1
best_val_f1 = 0
best_test_f1 = 0
best_epoch = -1
best_val_epoch = -1
plt.figure(num=1)
plt.figure(num=2)
plt.figure(num=3)
t = np.linspace(0, opt.epochs, opt.epochs)
epoch_test_loss = 0 * t
epoch_test_corrects = 0 * t
epoch_test_precisions = 0 * t
epoch_test_recalls = 0 * t
epoch_test_f1scores = 0 * t
epoch_test_learning_rate = 0 * t

epoch_val_loss = 0 * t
epoch_val_corrects = 0 * t
epoch_val_precisions = 0 * t
epoch_val_recalls = 0 * t
epoch_val_f1scores = 0 * t
epoch_val_learning_rate = 0 * t

epoch_train_loss = 0 * t
epoch_train_corrects = 0 * t
epoch_train_precisions = 0 * t
epoch_train_recalls = 0 * t
epoch_train_f1scores = 0 * t
epoch_train_learning_rate = 0 * t



epoch_list = [epoch_train_loss, epoch_train_corrects, epoch_train_precisions, epoch_train_recalls, epoch_train_f1scores, epoch_train_learning_rate]
epoch_loss,epoch_corrects, epoch_precisions, epoch_recalls, epoch_f1scores, epoch_learning_rate = epoch_list

epoch_val_list = [epoch_val_loss, epoch_val_corrects, epoch_val_precisions, epoch_val_recalls, epoch_val_f1scores, epoch_val_learning_rate]
epoch_val_loss, epoch_val_corrects, epoch_val_precisions, epoch_val_recalls, epoch_val_f1scores, epoch_val_learning_rate = epoch_val_list

epoch_test_list = [epoch_test_loss, epoch_test_corrects, epoch_test_precisions, epoch_test_recalls, epoch_test_f1scores, epoch_test_learning_rate]
epoch_test_loss, epoch_test_corrects, epoch_test_precisions, epoch_test_recalls, epoch_test_f1scores, epoch_test_learning_rate = epoch_test_list

for epoch in range(start_epoch+1,opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()
    test_metrics = initialize_metrics()
    exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=opt.learning_rate, epoch=epoch, warmup_epoch=5, max_epoch=opt.epochs)
    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        optimizer.zero_grad()
        cd_preds = model(batch_img1, batch_img2)
        cd_loss = criterion(cd_preds, labels)
        loss = cd_loss
        loss.backward()
        optimizer.step()

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1)

        # train_metrics = set_metrics(train_metrics,
                                    # cd_loss,
                                    # cd_corrects,
                                    # cd_train_report,
                                    # scheduler.get_last_lr())
                                
        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    exp_scheduler)
                                    
        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)      
        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    #scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))
    
    logging.info('updata the model')
    metadata['train_metrics'] = mean_train_metrics
   
    
    
    
    
    #Chart
    #print("mean_train_metrics['cd_losses']:",mean_train_metrics['cd_losses'])
    epoch_loss[epoch] = mean_train_metrics['cd_losses']
    epoch_corrects[epoch] = mean_train_metrics['cd_corrects']
    epoch_precisions[epoch] = mean_train_metrics['cd_precisions']
    epoch_recalls[epoch] = mean_train_metrics['cd_recalls']
    epoch_f1scores[epoch] = mean_train_metrics['cd_f1scores']
    epoch_learning_rate[epoch] = mean_train_metrics['learning_rate']

    
    plt.figure(num=1)
    plt.clf() 
    l1_1, = plt.plot(t[:epoch+1], epoch_f1scores[:epoch+1], label='Train f1scores')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_f1scores[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_f1scores[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_f1scores[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_f1scores[max_indx]),xy=(t[max_indx],epoch_f1scores[max_indx]))
    plt.title('F1-epoch')
    plt.savefig('./chart/train/Train_F1-epoch.png', bbox_inches='tight')
   
   
   
    plt.figure(num=2)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_recalls[:epoch+1], label='Train recalls')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('recall-epoch')
    plt.savefig('./chart/train/Train_recall-epoch.png', bbox_inches='tight')
    
    plt.figure(num=3)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_precisions[:epoch+1], label='Train precisions')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('precisions-epoch')
    plt.savefig('./chart/train/Train_precisions-epoch.png', bbox_inches='tight')
    
    plt.figure(num=4)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_loss[:epoch+1], label='Train loss')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('Loss')
    plt.savefig('./chart/train/Train_Loss-epoch.png', bbox_inches='tight')
    
    plt.figure(num=5)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_learning_rate[:epoch+1], label='Train learning_rate')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('Loss')
    plt.savefig('./chart/train/Train_learning_rate-epoch.png', bbox_inches='tight')
   
    # Save model and log
    
    with open('./tmp/train/metadata_train_epoch_' + str(epoch) + '.json', 'w') as fout:
        json.dump(metadata, fout)


    """
    Begin Validation
    """
    model.eval()
    batch_iter1 = 0
    tbar1 = tqdm(val_loader)
    Precision_val, Recall_val, F1_val, IoU_val, ACC_val = metrics_calculation(opt, model, batch_iter1, tbar1, epoch, 'val')
        
    epoch_val_loss[epoch] = IoU_val
    epoch_val_corrects[epoch] = ACC_val
    epoch_val_precisions[epoch] = Precision_val
    epoch_val_recalls[epoch] = Recall_val
    epoch_val_f1scores[epoch] = F1_val
    metircs_val = "IoU = "+ str(IoU_val) + ", " + "ACC = "+ str(ACC_val) + ", " + "Precision = " + str(Precision_val)+ ", " +"Recall ="+ str(Recall_val)+ ", " +"F1_score ="+ str(F1_val)
    metadata_val["metrics"] = metircs_val
    logging.info("Val_EPOCH {} Val METRICS ".format(epoch)+metircs_val)
    

    with open('./tmp/val/metadata_val_epoch_' + str(epoch) + '.json', 'w') as fout:
        json.dump(metadata_val, fout)
    if (F1_val > best_val_f1):
        torch.save(model, './tmp/best_checkpoint_epoch.pt')
        best_val_f1 = F1_val
        best_val_epoch = epoch
        with open('./tmp/metadata_best_checkpoint_epoch.json', 'w') as fout:
            metadata_val['epoch'] = str(epoch)
            json.dump(metadata_val, fout)


    
    plt.figure(num=1)
    plt.clf() 
    l1_1, = plt.plot(t[:epoch+1], epoch_val_f1scores[:epoch+1], label='val f1scores')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_val_f1scores[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_val_f1scores[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_val_f1scores[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_val_f1scores[max_indx]),xy=(t[max_indx],epoch_val_f1scores[max_indx]))
    plt.title('F1-epoch')
    plt.savefig('./chart/val/Val_F1-epoch.png', bbox_inches='tight')
   
   
   
    plt.figure(num=2)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_val_recalls[:epoch+1], label='val recalls')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('recall-epoch')
    plt.savefig('./chart/val/Val_recall-epoch.png', bbox_inches='tight')
    
    plt.figure(num=3)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_val_precisions[:epoch+1], label='val precisions')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('precisions-epoch')
    plt.savefig('./chart/val/Val_precisions-epoch.png', bbox_inches='tight')
    
    plt.figure(num=4)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_val_loss[:epoch+1], label='val loss')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('IoU')
    plt.savefig('./chart/val/Val_IoU-epoch.png', bbox_inches='tight')
    
    # plt.figure(num=5)
    # plt.clf()
    # l1_1, = plt.plot(t[:epoch+1], epoch_val_learning_rate[:epoch+1], label='val learning_rate')
    # plt.legend(handles=[l1_1])
    # plt.grid()
    # plt.gcf().gca().set_xlim(left = 0)
    # plt.title('Loss')
    # plt.savefig('./chart/val/Val_learning_rate-epoch.png', bbox_inches='tight')
    
    
    
    
    
    """
    Begin test
    """
    model.eval()
    batch_iter2 = 0
    tbar2 = tqdm(test_loader)
    Precision, Recall, F1, IoU, ACC = metrics_calculation(opt, model, batch_iter2, tbar2, epoch, 'test')
    
    epoch_test_loss[epoch] = IoU
    epoch_test_corrects[epoch] = ACC
    epoch_test_precisions[epoch] = Precision
    epoch_test_recalls[epoch] = Recall
    epoch_test_f1scores[epoch] = F1
    metircs_test = "IoU = "+ str(IoU) + ", " + "ACC = "+ str(ACC) + ", " + "Precision = " + str(Precision)+ ", " +"Recall ="+ str(Recall)+ ", " +"F1_score ="+ str(F1)
    metadata_test["metrics"] = metircs_test
    logging.info("Test_EPOCH {} Test METRICS ".format(epoch)+metircs_test)
   
   
    with open('./tmp/test/metadata_test_epoch_' + str(epoch) + '.json', 'w') as fout:
        json.dump(metadata_test, fout)
    if (F1 > best_test_f1):
        #torch.save(model, './tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        torch.save(model, './tmp/best_test_checkpoint_epoch.pt')
        best_test_f1 = F1
        best_epoch = epoch
        with open('./tmp/metadata_test_best_checkpoint_epoch.json', 'w') as fout:
            metadata_test['epoch'] = str(epoch)
            json.dump(metadata_test, fout)
    # comet.log_asset(upload_metadata_file_path)
    #best_metrics = mean_val_metrics 
    logging.info("The current best_val_epoch {} Test ".format(best_val_epoch)+"F1_score: "+str(best_val_f1))
    logging.info("The current best epoch {} Test ".format(best_epoch)+"F1_score: "+str(best_test_f1))
    
    plt.figure(num=1)
    plt.clf() 
    l1_1, = plt.plot(t[:epoch+1], epoch_test_f1scores[:epoch+1], label='test f1scores')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_test_f1scores[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_test_f1scores[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_test_f1scores[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_test_f1scores[max_indx]),xy=(t[max_indx],epoch_test_f1scores[max_indx]))
    plt.title('F1-epoch')
    plt.savefig('./chart/test/Test_F1-epoch.png', bbox_inches='tight')
   
   
   
    plt.figure(num=2)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_test_recalls[:epoch+1], label='test recalls')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('recall-epoch')
    plt.savefig('./chart/test/Test_recall-epoch.png', bbox_inches='tight')
    
    plt.figure(num=3)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_test_precisions[:epoch+1], label='test precisions')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('precisions-epoch')
    plt.savefig('./chart/test/Test_precisions-epoch.png', bbox_inches='tight')
    
    plt.figure(num=4)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_test_loss[:epoch+1], label='test IoU')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('IoU-epoch')
    plt.savefig('./chart/test/Test_IoU-epoch.png', bbox_inches='tight')
    
    # plt.figure(num=5)
    # plt.clf()
    # l1_1, = plt.plot(t[:epoch+1], epoch_test_learning_rate[:epoch+1], label='test learning_rate')
    # plt.legend(handles=[l1_1])
    # plt.grid()
    # plt.gcf().gca().set_xlim(left = 0)
    # plt.title('Loss')
    # plt.savefig('./chart/test/Test_learning_rate-epoch.png', bbox_inches='tight')
    
    print('An epoch finished.')
    print()
writer.close()  # close tensor board
print('Done!')

