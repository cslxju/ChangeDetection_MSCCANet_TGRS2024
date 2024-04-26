import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.parser import get_parser_with_args
import sys
import os
import re
import time
import json
# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

gpu_num = input("GPU_ID:")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)
path_in = './tmp'
files = os.listdir(path_in)
b = []
for f in files:
    if(f[-1]=='t'):
        f = f[17:-3]
        f = int(f)
        b.append(f)
b.sort()
print(b)

model_number = input("输入起始模型号：")
model_number_index = b.index(int(model_number))
model_nums = input("测试模型数量：")
data_dict = {}
model_list = b[model_number_index:model_number_index+int(model_nums)]
print("待测试的模型为：")
for i in model_list:
    print("checkpoint_epoch_"+str(i))
print()
 
max_P = 0.0
max_R = 0.0
max_F1 = 0.0
Best_model_P = ""
Best_model_R = ""
Best_model_F1 = ""

time_tuple = time.localtime(time.time())
tim = "{}年{}月{}日{}点{}分{}秒".format(time_tuple[0],time_tuple[1],time_tuple[2],time_tuple[3],time_tuple[4],time_tuple[5])
path1 = path_in+'/eval_record/'+tim
if not os.path.exists(path1):
    os.makedirs(path1)
print(tim)
parser, metadata = get_parser_with_args()

for i in model_list:
    
    
    path = path_in+'/checkpoint_epoch_'+str(i)+'.pt'   # the path of the model
    model_name = "checkpoint_epoch_"+str(i)
    print(model_name+" is testing...")
    model = torch.load(path)
    
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    model.eval()

    with torch.no_grad():
        tbar = tqdm(test_loader)
        for batch_img1, batch_img2, labels in tbar:

            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            cd_preds = model(batch_img1, batch_img2)
            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)
            tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                            cd_preds.data.cpu().numpy().flatten(),labels=[0,1]).ravel()
            # zip_file = confusion_matrix(labels.data.cpu().numpy().flatten(),
                            # cd_preds.data.cpu().numpy().flatten()).ravel()
            # #print("zip_file.size()",zip_file.size)
            # if(zip_file.size==1):
                # i = labels.size()[0]
                # for j in range(i):
                    # if(labels[j,0,0] == 0):
                        # cd_preds[j,0,0] = 1
                    # elif(labels[j,0,0] == 1):
                        # cd_preds[j,0,0] = 0
                # zip_file = confusion_matrix(labels.data.cpu().numpy().flatten(),
                            # cd_preds.data.cpu().numpy().flatten()).ravel()

            # tn = zip_file[0]
            # fp = zip_file[1]
            # fn = zip_file[2]
            # tp = zip_file[3]
            
            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    print('tn:',tn)
    print('fp:',fp)
    print('fn:',fn)
    print('tp:',tp)
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    IoU = tp / (tp + fp + fn) 
    Acc = (tp + tn) / (tp + fp + tn + fn)
    if(P>max_P):
        max_P = P
        Best_model_P = model_name
    if(R>max_R):
        max_R = R 
        Best_model_R = model_name
    if(F1>max_F1):
        max_F1 = F1
        Best_model_F1 = model_name
    data_dict[model_name] = 'Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1)
    data_tmp = {}
    data_tmp['tn'] = str(tn)
    data_tmp['fp'] = str(fp)
    data_tmp['fn'] = str(fn)
    data_tmp['tp'] = str(tp)
    data_tmp['Precision'] = P
    data_tmp['Recall'] = R
    data_tmp['F1_scores'] = F1
    data_tmp['IoU'] = IoU
    data_tmp['Acc'] = Acc
    
    with open(path1+'/' + model_name + '.json', 'w') as fout:
        json.dump(metadata, fout,indent=3)
        json.dump(data_tmp, fout,indent=3)
    print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))
print("Best_model_P: ",Best_model_P)
print(data_dict[Best_model_P])
print()
print("Best_model_R: ",Best_model_R)
print(data_dict[Best_model_R])
print()
print("Best_model_F1: ",Best_model_F1)
print(data_dict[Best_model_F1])

best_data_p = {}
best_data_r = {}
best_data_f = {}

best_data_p['Best_model_P'] = Best_model_P+":  "+data_dict[Best_model_P]+"    ".replace('\n','  ')
best_data_r['Best_model_R'] = Best_model_R+":  "+data_dict[Best_model_P]+"    ".replace('\n','  ')
best_data_f['Best_model_F1'] = Best_model_F1+":  "+data_dict[Best_model_F1]+"    ".replace('\n','  ')
# P = 0.9555685
# R = 0.98666
# F1 = 0.962563
# metadat = {}
# metada =   {}
# str1 = 'Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1).replace('\n','    ')
# metadata['Best_model_P'] = P
# metadat['Best_model_R'] = R
# metada['Best_model_F1'] = F1
with open(path1+'/' + 'Best_Resout' + '.json', 'w') as fout:
        json.dump(best_data_p, fout,indent=3)
        json.dump(best_data_r, fout,indent=3)
        json.dump(best_data_f, fout,indent=3)
  

