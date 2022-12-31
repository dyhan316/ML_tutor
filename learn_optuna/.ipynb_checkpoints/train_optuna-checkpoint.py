##

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import random 
from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing
from tqdm import tqdm
import time
import glob

from envs.dataset import ImageDataset, get_minority_class
from envs.custom_optim import SAM, LAMB, LARS
import csv

from envs.utils import get_scheduler, load, Acc_EarlyStopping
from envs.model import CNN3D, VGG3D, resnet3D50, resnet3D101, resnet3D152
#import models.densenet3d as densenet3d #model script
from monai.networks.nets import DenseNet
from sklearn.metrics import  confusion_matrix, roc_auc_score

from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, ToTensor

#optuna
import optuna
import sys
import logging


## ========= Data Preprocessing ========= ##
def data_preprocessing_CycleGAN(args):
    # Data Read
    # getting image file names (subject ID + '.npy') as list
    data_dir = args.data_dir
    os.chdir(data_dir)
    
    subjects = sorted(os.listdir(data_dir))

    subject_data = pd.read_csv(args.subject_info_file)
    subject_data = subject_data.loc[:, ['subjectkey', args.target]]
    subject_data = subject_data.dropna(axis=0).reset_index(drop=True)

    # site classification
    # analyze with whole dataset
    if args.setting == 'site': 
        discovery = subject_data[subject_data[args.target] == 'DISCOVERY MR750']
        prisma = subject_data[subject_data[args.target] == 'Prisma']

        discovery_files = [os.path.join(data_dir,key+'.nii.gz') for key in list(discovery['subjectkey']) if key+'.nii.gz' in subjects]
        discovery_subjectkeys = list(discovery['subjectkey'])
        prisma_files = [os.path.join(data_dir,key+'.nii.gz') for key in list(prisma['subjectkey']) if key+'.nii.gz' in subjects]
        prisma_subjectkeys = list(prisma['subjectkey'])

        input_a = [ (i, 0) for i in discovery_files]
        input_b = [ (i, 1) for i in prisma_files]
        imageFiles_labels = input_a + input_b
        print('numbers of input_a(discovery):', len(input_a))
        print('numbers of input_b(prisma_files):', len(input_b))

    # meta data classification
    # analyze with a cyclegan 'result/test' folder      

    else:
        #label0: discovery(GE), label1: prisma(Siemens)
        #결과물로 분석시

        a2b = os.listdir(os.path.join(data_dir,'a2b'))
        input_a = [ os.path.join(data_dir,'a2b',i) for i in a2b if 'input_a' in i]
        output_b = [ os.path.join(data_dir,'a2b',i) for i in a2b if 'output_b' in i ]

        b2a = os.listdir(os.path.join(data_dir,'b2a'))
        input_b = [ os.path.join(data_dir,'b2a',i) for i in b2a if 'input_b' in i]
        output_a = [ os.path.join(data_dir,'b2a',i) for i in b2a if 'output_a' in i ]

        if args.setting=="ab":
            # getting subject ID and target variables as sorted list
            print('input_a + input_b')
            print('numbers of input_a(discovery):', len(input_a))
            print('numbers of input_b(prisma):', len(input_b))
            imageFiles= input_a + input_b
        elif args.setting == "aa_":
            print('input_a + b_harmonized_to_a')
            print('numbers of input_a(discovery):', len(input_a))
            print('numbers of output_a(prisma->discovery):', len(output_a))
            imageFiles= input_a + output_a
        elif args.setting == "bb_":
            print('input_b + a_harmonized_to_b')
            print('numbers of input_b(prisma):', len(input_b))
            print('numbers of output_b(discovery->prisma):', len(output_b))
            imageFiles = input_b + output_b

        
        # contain someone who has the labels
        subjects = list(subject_data['subjectkey']) 

        if args.task_type == "classification":
            #label encoding for target variable
            le = preprocessing.LabelEncoder()
            le.fit(list(set(subject_data[args.target])))

            label_before = le.classes_
            label_after = le.transform(label_before)
            label_description = {i:j for i, j in zip(label_before, label_after)}
            print(label_description)
            subject_data[args.target] = le.transform(subject_data[args.target])

        elif args.task_type == "regression":
            #standarize continuous labels to use MSE loss
            mean = np.mean(subject_data[args.target],axis=0)
            std = np.std(subject_data[args.target],axis=0)
            subject_data[args.target] = ((subject_data[args.target]-mean)/std).astype(np.float64)

        imageFiles_labels = [(file,subject_data[subject_data['subjectkey']==file[file.find('ND'):file.find('ND')+15]][args.target].item()) for file in imageFiles if file[file.find('ND'):file.find('ND')+15] in subjects]
        labels = [label for key, label in imageFiles_labels]
        

        # print numbers of each label
        print( {i: labels.count(i) for i in set(labels)} )
        
        return imageFiles_labels
    


def data_preprocessing_Combat(args):
    # Data Read
    # getting image file names (subject ID + '.npy') as list
    data_dir = args.data_dir
    os.chdir(data_dir)
    
    subjects = sorted(glob.glob('*_harmonized.nii.gz'))
    
    subject_data = pd.read_csv(args.subject_info_file)
    subject_data = subject_data.loc[:, ['subjectkey', args.target]]
    subject_data = subject_data.dropna(axis=0).reset_index(drop=True)

    # site classifier 학습
    # analyze with whole dataset
    if args.setting == 'site': 
        subject_data = subject_data.loc[:, ['subjectkey', args.target]]
        subject_data = subject_data.dropna(axis=0).reset_index(drop=True)
        discovery = subject_data[subject_data[args.target] == 'DISCOVERY MR750']
        prisma = subject_data[subject_data[args.target] == 'Prisma']

        discovery_files = [os.path.join(data_dir,key+'.nii.gz') for key in list(discovery['subjectkey']) if key+'_harmonized.nii.gz' in subjects]
        discovery_subjectkeys = list(discovery['subjectkey'])
        prisma_files = [os.path.join(data_dir,key+'.nii.gz') for key in list(prisma['subjectkey']) if key+'_harmonized.nii.gz' in subjects]
        prisma_subjectkeys = list(prisma['subjectkey'])

        input_a = [ (i, 0) for i in discovery_files]
        input_b = [ (i, 1) for i in prisma_files]
        imageFiles_labels = input_a + input_b
        print('numbers of input_a(discovery):', len(input_a))
        print('numbers of input_b(prisma_files):', len(input_b))
        
    else:
        
        imageFiles = [os.path.join(data_dir,file) for file in subjects]
        # contain someone who has the labels
        subjects = list(subject_data['subjectkey']) 

        if args.task_type == "classification":
            #label encoding for target variable
            le = preprocessing.LabelEncoder()
            le.fit(list(set(subject_data[args.target])))

            label_before = le.classes_
            label_after = le.transform(label_before)
            label_description = {i:j for i, j in zip(label_before, label_after)}
            print(label_description)
            subject_data[args.target] = le.transform(subject_data[args.target])

        elif args.task_type == "regression":
            #standarize continuous labels to use MSE loss
            mean = np.mean(subject_data[args.target],axis=0)
            std = np.std(subject_data[args.target],axis=0)
            subject_data[args.target] = ((subject_data[args.target]-mean)/std).astype(np.float64)

        imageFiles_labels = [(file,subject_data[subject_data['subjectkey']==file[file.find('ND'):file.find('ND')+15]][args.target].item()) for file in imageFiles if file[file.find('ND'):file.find('ND')+15] in subjects]
        labels = [label for key, label in imageFiles_labels]


        # print numbers of each label
        print( {i: labels.count(i) for i in set(labels)} )

    return imageFiles_labels    
    
    
## Dataset
# defining train,val, test set splitting function
def partition(imageFiles_labels,args):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        images.append(image)
        labels.append(label)

    resize = args.resize
    train_transform = Compose([ScaleIntensity(minv=-1.0, maxv=1.0),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(minv=-1.0, maxv=1.0),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(minv=-1.0, maxv=1.0),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    num_total = len(images) # len(images[:100])
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    #print(num_train)
    num_val = int(num_total*args.val_size)
    #print(num_val)
    num_test = int(num_total*args.test_size)
    #print(num_test)

    images_train = images[:num_train]
    labels_train = labels[:num_train]

    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    images_test = images[num_total-num_test:]
    labels_test = labels[num_total-num_test:]

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    return partition


## Dataset
# defining train,val, test set splitting function
# k-fold cross validation
def partition_kfold(imageFiles_labels,args):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        images.append(image)
        labels.append(label)
        
    images = np.array(images)
    labels = np.array(labels)
    
    partitions = {}
    skf = StratifiedKFold(n_splits=args.k_folds)
    i=0
    for train_index, test_index in skf.split(images, labels):
        #print("TRAIN:", train_index, "TEST:", test_index)
        images_train, images_test = images[train_index], images[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        
        # train의 일부를 valid로 구분
        images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=0.2, random_state=42)
        
        resize = args.resize
        train_transform = Compose([ScaleIntensity(minv=-1.0, maxv=1.0),
                                   AddChannel(),
                                   Resize(resize),
                                  ToTensor()])

        val_transform = Compose([ScaleIntensity(minv=-1.0, maxv=1.0),
                                   AddChannel(),
                                   Resize(resize),
                                  ToTensor()])

        test_transform = Compose([ScaleIntensity(minv=-1.0, maxv=1.0),
                                   AddChannel(),
                                   Resize(resize),
                                  ToTensor()])

        # num_total = len(images)
        # num_train = int(num_total*(1 - args.val_size - args.test_size))
        #print(num_train)
        # num_val = int(num_total*args.val_size)
        #print(num_val)
        # num_test = int(num_total*args.test_size)
        #print(num_test)

#         images_train = images[:num_train]
#         labels_train = labels[:num_train]

#         images_val = images[num_train:num_train+num_val]
#         labels_val = labels[num_train:num_train+num_val]

#         images_test = images[num_total-num_test:]
#         labels_test = labels[num_total-num_test:]

        train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
        val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
        test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

        partition = {}
        partition['train'] = train_set
        partition['val'] = val_set
        partition['test'] = test_set
        
        partitions[f'{i}'] = partition
        i+=1

    return partitions


## ========= Experiment =============== ##
def optuna_tuning(partition,args): #in_channels,out_dim
    
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    #optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Optuna settings
    
    storage = optuna.storages.RDBStorage(
        url = args.study_storage,
        heartbeat_interval = 60, # 60초마다 heartbeat를 보낸다
        grace_period = 120, # 2분동안 응답이 없으면 상태를 전환시킨다
        # failed_trial_callback=RetryFailedTrialCallback(max_retry=1)
        )
    
    #fix sampler with respect to random seed
    study = optuna.create_study(study_name = args.study_name, 
                                sampler = optuna.samplers.RandomSampler(seed = args.seed), #하이퍼 파라미터를 샘플링하는 방법
                                pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=args.warmup, interval_steps=1),
                                storage = storage,                                
                                load_if_exists = True,
                                direction = 'maximize')
    
    def objective(trial):
        #hyperparameters
        args.lr = trial.suggest_float("learning_rate", 1e-4, 1e-3)
        # args.l2 = trial.suggest_float("l2", 1e-4, 5*1e-4, log=True)
        args.lr_policy = trial.suggest_categorical("lr_policy",["SGDR","step","plateau"])
        args.gamma = trial.suggest_float("gamma", 0.1, 0.5, step=0.1)
        args.network = trial.suggest_categorical("network", ["3DCNN","resnet3D50"])  #VGG is not working, "resnet3D101",'DenseNet121' is too big
        
        
    #         # Integer parameter
    #         num_layers = trial.suggest_int("num_layers", 1, 3)

#         # Integer parameter (log)
#         num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

#         # Integer parameter (discretized)
#         num_units = trial.suggest_int("num_units", 10, 100, step=5)

#         # Floating point parameter
#         dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

#         # Floating point parameter (log)
#         learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

#         # Floating point parameter (discretized)
#         drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
        
        
        if args.task_type == 'regression':
            args.out_dim = 1
        if args.task_type == 'classification':     
            criterion = nn.CrossEntropyLoss()
        elif args.task_type == 'regression': 
            criterion = nn.MSELoss()
        
        
        if args.network == "3DCNN":
            net = CNN3D(in_channels=args.in_channels,
                          out_dim=args.out_dim)
        elif args.network.startswith("VGG"):
            net = VGG3D(model_code=args.network,
                    in_channels=args.in_channels,
                    out_dim=args.out_dim)
        elif args.network == "resnet3D50":
            net = resnet3D50(num_classes=args.out_dim,in_channels=args.in_channels)
        elif args.network == "resnet3D101":
            net = resnet3D101(num_classes=args.out_dim,in_channels=args.in_channels)
        elif args.network == "resnet3D152":
            net = resnet3D152(num_classes=args.out_dim,in_channels=args.in_channels)
        elif args.network == "DenseNet121":
            net = DenseNet(spatial_dims=3, in_channels = args.in_channels, out_channels=args.out_dim, init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.0)
            
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'RMSprop':
            optimizer = torch.optim.RMSprop(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'SAM':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(net.parameters(),base_optimizer,lr=args.lr,momentum=0.9)
        elif args.optim == 'LARS':
            optimizer = LARS(net.parameters(), lr=args.lr, momentum=0.9)
        elif args.optim == 'LAMB':
            optimizer = LAMB(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)   
        else:
            raise ValueError('In-valid optimizer choice')
        

        net = nn.DataParallel(net,device_ids=args.gpus_list) # 서버 상황에 따라 선택해야함
        net.to(f'cuda:{net.device_ids[0]}')
        #net = nn.DataParallel(net)
        #net.cuda()

        scheduler = get_scheduler(optimizer, args)

        result = {}
        st_epoch = 0
        
        lrs = []
                
        best_acc = 0
        for epoch in tqdm(range(st_epoch+1, args.epoch+1)):
            ts = time.time()
            #train
            net, train_loss, train_acc, train_auroc = train(net,partition,optimizer,criterion,args)
            te = time.time()

            #validation
            val_loss, val_acc, val_auroc = validate(net,partition,criterion,args)
            print(
                'Epoch {}, ACC or R^2(train/val): {:2.2f}/{:2.2f}, AUC(train/val): {:2.2f}/{:2.2f}, Loss(train/val): {:2.2f}/{:2.2f}. Lr: {:f}. Took {:0.2f} sec'.format(
                    epoch, train_acc, val_acc, train_auroc, val_auroc, train_loss, val_loss, optimizer.param_groups[0]["lr"], te - ts))

            if val_acc > best_acc:
                best_acc = val_acc

            trial.set_user_attr("val_loss", val_loss)
            trial.set_user_attr("val_AUC", val_auroc)

            trial.report(val_acc, epoch) # report current performence

            # Prune if sub-optimal
            if trial.should_prune():
                raise optuna.TrialPruned()


            if scheduler is not None:
                if args.lr_policy == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            return best_acc # return the best performance 
        
    study.optimize(objective, n_trials=args.n_trials) #objective는 위 function # trial을 몇 번 돌리는가? #보통 100~200번
    # butler.logger.info('saving the study: {}'.format(storage_addr))   
    
    
    print("Best trial:")
    print('best trial info: {}'.format(study.best_trial))
    print('best parameters: {}'.format(study.best_params))
    print('best metric: {}'.format(study.best_value))
    
    for key,value in study.best_params.items():
        setattr(args,key,value)
    return args

## ========= Train,Validate, and Test ========= ##
# define training step
def train(net,partition,optimizer,criterion,args):
    # scaler is for transforming loss value from 32bit to 16 bit  
    scaler = torch.cuda.amp.GradScaler()
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.train_batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             drop_last=True)

    net.train()

    correct = 0
    total = 0
    train_loss = 0.0
    labels = []
    preds = []
    probas = []
    r_squares = []

    if args.optim == 'SAM':
        for i, data in enumerate(trainloader,0):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            

            def closure():
                loss = criterion(net(image),label)
                loss.backward()
                return loss

            loss = criterion(net(image),label)
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = torch.max(net(image).data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    else:
        for i, data in enumerate(trainloader,0):

            optimizer.zero_grad()
            image, label = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            
            #change to float dtype float for continuous labels
            if args.task_type == "regression":
                label = label.to(f'cuda:{net.device_ids[0]}').float()
            elif args.task_type == "classification":
                label = label.to(f'cuda:{net.device_ids[0]}')
                
            #image = image.cuda()
            #label = label.cuda()
            output = net(image)
            loss = criterion(output.squeeze(),label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if args.task_type == 'classification':
                
                m = torch.nn.Softmax(dim=1)
                output = m(output)
                
                _, predicted = torch.max(output.data,1)
                labels.append(label.cpu().int().numpy())
                preds.append(predicted.cpu().int().numpy())
                probas.append(output.detach().cpu().numpy())
                
                total += label.size(0)
                correct += (predicted == label).sum().item()
                
            elif args.task_type == 'regression':
                y_var = torch.var(label)
                r_square = 1 - (loss / y_var)
                r_squares.append(r_square.item())
            
            


    train_loss = train_loss / len(trainloader)
    if args.task_type == 'classification':
        train_acc = 100 * correct / total
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        
        probas = np.concatenate(probas,axis=0)[:,1]

        
        train_auroc = roc_auc_score(labels,probas)
        return net, train_loss, train_acc, train_auroc

    elif args.task_type == 'regression':
        train_rsquare = np.mean(r_squares)
        return net, train_loss, train_rsquare, 0

# define validation step
def validate(net,partition,criterion,args):

    if args.val_equal_samples == True:
        val_images = partition['val'].image_files
        val_labels = partition['val'].labels
        val_transform = partition['val'].transform

        classes, minority_class, minority_cnt = get_minority_class(val_labels)
        total_indexes = []
        for label in classes:
            indexes = np.where(np.array(val_labels) == label)[0].tolist()
            total_indexes += random.sample(indexes, minority_cnt)
        image_files = np.array(val_images)[total_indexes]
        labels = np.array(val_labels)[total_indexes]
        controlled_dataset = ImageDataset(image_files=image_files, labels=labels, transform=val_transform)
        valloader = torch.utils.data.DataLoader(controlled_dataset,
                                                batch_size=args.val_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers)

    else:
        valloader = torch.utils.data.DataLoader(partition['val'],
                                           batch_size=args.val_batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers)

    net.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    labels = []
    preds = []
    probas = []
    r_squares = []

    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image, label = data
            image = image.to(f'cuda:{net.device_ids[0]}')
            
            #change to float dtype float for continuous labels
            if args.task_type == "regression":
                label = label.to(f'cuda:{net.device_ids[0]}').float()
            elif args.task_type == "classification":
                label = label.to(f'cuda:{net.device_ids[0]}')
                
                
            #image = image.cuda()
            #label = label.cuda()
            output = net(image)

            loss = criterion(output.squeeze(),label)

            val_loss += loss.item()
            
            if args.task_type == 'classification':
                m = torch.nn.Softmax(dim=1)
                output = m(output)
                
                _, predicted = torch.max(output.data,1)
                labels.append(label.cpu().int().numpy())
                preds.append(predicted.cpu().int().numpy())
                probas.append(output.detach().cpu().numpy())
                
                total += label.size(0)
                correct += (predicted == label).sum().item()
                
            elif args.task_type == 'regression':
                y_var = torch.var(label)
                r_square = 1 - (loss / y_var)
                r_squares.append(r_square.item())
                
        val_loss = val_loss / len(valloader)

        if args.task_type == 'classification':
            val_acc = 100 * correct / total
            labels = np.concatenate(labels)
            preds = np.concatenate(preds)
            probas = np.concatenate(probas,axis=0)[:,1]
            
            val_auroc = roc_auc_score(labels,probas)
        
            return val_loss, val_acc, val_auroc

        elif args.task_type == 'regression':
            val_rsquare = np.mean(r_squares)
            return val_loss, val_rsquare, 0

# define test step
def test(net,partition,criterion,args):
    print('test starting')
    if args.val_equal_samples == True:
        test_images = partition['test'].image_files
        test_labels = partition['test'].labels
        test_transform = partition['test'].transform

        classes, minority_class, minority_cnt = get_minority_class(test_labels)
        total_indexes = []
        for label in classes:
            indexes = np.where(np.array(test_labels) == label)[0].tolist()
            total_indexes += random.sample(indexes, minority_cnt)
        image_files = np.array(test_images)[total_indexes]
        labels = np.array(test_labels)[total_indexes]
        controlled_dataset = ImageDataset(image_files=image_files, labels=labels, transform=test_transform)
        testloader = torch.utils.data.DataLoader(controlled_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers)
    else:
        testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers)

    net.eval()

    correct = 0
    total = 0
    labels = []
    preds = []
    probas = []
    r_squares = []

    for i, data in enumerate(testloader,0):
        image, label = data
        image = image.to(f'cuda:{net.device_ids[0]}')
        #change to float dtype float for continuous labels
        if args.task_type == "regression":
            label = label.to(f'cuda:{net.device_ids[0]}').float()
        elif args.task_type == "classification":
            label = label.to(f'cuda:{net.device_ids[0]}')
        #image = image.cuda()
        #label = label.cuda()
        output = net(image)

        loss = criterion(output.squeeze(),label)
        if args.task_type == 'classification':
            
            m = torch.nn.Softmax(dim=1)
            output = m(output)
            
            _, predicted = torch.max(output.data,1)
            
            labels.append(label.cpu().int().numpy())
            preds.append(predicted.cpu().int().numpy())
            probas.append(output.detach().cpu().numpy())
        
            total += label.size(0)
            correct += (predicted == label).sum().item()

        elif args.task_type == 'regression':
            y_var = torch.var(label)
            r_square = 1 - (loss / y_var)
            r_squares.append(r_square.item())

    if args.task_type == 'classification':
        test_acc = 100 * correct / total
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        probas = np.concatenate(probas,axis=0)[:,1]
        test_auroc = roc_auc_score(labels,probas)
        print('test_acc:', test_acc)
        print('test_auc:', test_auroc)
        print('confusion_matrix: \n', confusion_matrix(y_true=labels,y_pred=preds))
        return test_acc, test_auroc

    elif args.task_type == 'regression':
        test_rsquare = r_square.mean() 
        return test_rsquare, loss.item()

#===============================================

# train with best parameters
def experiment(partition,args):
    if args.task_type == 'regression':
        args.out_dim = 1

    if args.network == "3DCNN":
        net = CNN3D(in_channels=args.in_channels,
                      out_dim=args.out_dim)
    elif args.network.startswith("VGG"):
        net = VGG3D(model_code=args.network,
                in_channels=args.in_channels,
                out_dim=args.out_dim)
    elif args.network == "resnet3D50":
        net = resnet3D50(num_classes=args.out_dim)
    elif args.network == "resnet3D101":
        net = resnet3D101(num_classes=args.out_dim)
    elif args.network == "resnet3D152":
        net = resnet3D152(num_classes=args.out_dim)
    elif args.network == "DenseNet121":
        net = DenseNet(spatial_dims=3, in_channels = args.in_channels, out_channels=args.out_dim, init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.0)

    net = nn.DataParallel(net,device_ids=args.gpus_list) # 서버 상황에 따라 선택해야함
    net.to(f'cuda:{net.device_ids[0]}')
    #net = nn.DataParallel(net)
    #net.cuda()

    if args.task_type == 'classification':     
        criterion = nn.CrossEntropyLoss()
    elif args.task_type == 'regression': 
        criterion = nn.MSELoss()


    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optim == 'SAM':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(net.parameters(),base_optimizer,lr=args.lr,momentum=0.9)
    elif args.optim == 'LARS':
        optimizer = LARS(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'LAMB':
        optimizer = LAMB(net.parameters(), lr=args.lr, weight_decay=args.l2)   
    else:
        raise ValueError('In-valid optimizer choice')

    scheduler = get_scheduler(optimizer, args) # None

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []


    early_stopping = Acc_EarlyStopping(patience=args.patience, verbose=True, path=args.ckpt_dir, delta=0.02)

    # Summary writer for tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(args.log_dir))
    writer_val = SummaryWriter(log_dir=os.path.join(args.log_dir))

    result = {}
    st_epoch = 0

    lrs = []

    if args.test_only == "off":
        if (args.train_continue == "on"):
            net, optimizer, st_epoch = load(ckpt_dir=args.ckpt_dir, net=net, optim=optimizer)

        for epoch in tqdm(range(st_epoch+1, args.epoch+1)):         

            ts = time.time()

            #train
            net, train_loss, train_acc, train_auroc = train(net,partition,optimizer,criterion,args)

            #tensorboard
            writer_train.add_scalar('loss', train_loss, epoch)
            writer_train.add_scalar('acc', train_acc, epoch)
            te = time.time()

            #validation
            val_loss, val_acc, val_auroc = validate(net,partition,criterion,args)
            writer_val.add_scalar('loss', val_loss, epoch)
            writer_val.add_scalar('acc', val_acc, epoch)

            print(
                'Epoch {}, ACC or R^2(train/val): {:2.2f}/{:2.2f}, AUC(train/val): {:2.2f}/{:2.2f}, Loss(train/val): {:2.2f}/{:2.2f}. Lr: {:f}. Took {:0.2f} sec'.format(
                    epoch, train_acc, val_acc, train_auroc, val_auroc, train_loss, val_loss, optimizer.param_groups[0]["lr"], te - ts))

            early_stopping(val_acc, net, epoch, optimizer)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if scheduler is not None:
                writer_val.add_scalar('lr',optimizer.param_groups[0]["lr"], epoch)
                lrs.append(optimizer.param_groups[0]["lr"])
                if args.lr_policy == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
            # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
            if early_stopping.early_stop:
                print("Early stopping")
                break

        writer_train.close()
        writer_val.close()

        result['train_losses'] = train_losses
        result['train_accs'] = train_accs
        result['val_losses'] = val_losses
        result['val_accs'] = val_accs
        result['lr'] = lrs

    net, optim, st_epoch = load(ckpt_dir=args.ckpt_dir, net=net, optim=optimizer)
    test_acc, test_auc = test(net,partition,criterion,args)

    result['test_acc_or_R^2'] = test_acc
    result['test_auc_or_test_loss'] = test_auc

    return vars(args), result


# train with k-fold cross-validation
def experiment_kfold(partitions,args):
    if not os.path.exists(args.result_dir + 'kfold_results.csv'):
        with open(args.result_dir + 'kfold_results.csv', 'w',encoding='utf-8',newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['fold', "acc", "auc"])

    for fold, partition in partitions.items():
        if int(fold) < args.st_fold:
            continue

        assert int(fold) >= args.st_fold

        print(f'processing {fold} from total {args.k_folds} folds')

        if args.task_type == 'regression':
            args.out_dim = 1

        if args.network == "3DCNN":
            net = CNN3D(in_channels=args.in_channels,
                          out_dim=args.out_dim)
        elif args.network.startswith("VGG"):
            net = VGG3D(model_code=args.network,
                    in_channels=args.in_channels,
                    out_dim=args.out_dim)
        elif args.network == "resnet3D50":
            net = resnet3D50(num_classes=args.out_dim)
        elif args.network == "resnet3D101":
            net = resnet3D101(num_classes=args.out_dim)
        elif args.network == "resnet3D152":
            net = resnet3D152(num_classes=args.out_dim)
        elif args.network == "DenseNet121":
            net = DenseNet(spatial_dims=3, in_channels = args.in_channels, out_channels=args.out_dim, init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.0)

        net = nn.DataParallel(net,device_ids=args.gpus_list) # 서버 상황에 따라 선택해야함
        net.to(f'cuda:{net.device_ids[0]}')
        #net = nn.DataParallel(net)
        #net.cuda()

        if args.task_type == 'classification':     
            criterion = nn.CrossEntropyLoss()
        elif args.task_type == 'regression': 
            criterion = nn.MSELoss()


        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'RMSprop':
            optimizer = torch.optim.RMSprop(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.l2)
        elif args.optim == 'SAM':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(net.parameters(),base_optimizer,lr=args.lr,momentum=0.9)
        elif args.optim == 'LARS':
            optimizer = LARS(net.parameters(), lr=args.lr, momentum=0.9)
        elif args.optim == 'LAMB':
            optimizer = LAMB(net.parameters(), lr=args.lr, weight_decay=args.l2)   
        else:
            raise ValueError('In-valid optimizer choice')

        scheduler = get_scheduler(optimizer, args) # None

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []


        early_stopping = Acc_EarlyStopping(patience=args.patience, verbose=True, path=args.ckpt_dir, delta=0.02)

        # Summary writer for tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'+ str(fold)))
        writer_val = SummaryWriter(log_dir=os.path.join(args.log_dir, 'val' + str(fold)))

        result = {}
        st_epoch = 0

        lrs = []

        if args.test_only == "off":
            if (args.train_continue == "on"):
                net, optimizer, st_epoch = load(ckpt_dir=args.ckpt_dir, net=net, optim=optimizer)
                    
            for epoch in tqdm(range(st_epoch+1, args.epoch+1)):         

                ts = time.time()

                #train
                net, train_loss, train_acc, train_auroc = train(net,partition,optimizer,criterion,args)

                #tensorboard
                writer_train.add_scalar('loss', train_loss, epoch)
                writer_train.add_scalar('acc', train_acc, epoch)
                te = time.time()

                #validation
                val_loss, val_acc, val_auroc = validate(net,partition,criterion,args)
                writer_val.add_scalar('loss', val_loss, epoch)
                writer_val.add_scalar('acc', val_acc, epoch)

                print(
                    'Epoch {}, ACC or R^2(train/val): {:2.2f}/{:2.2f}, AUC(train/val): {:2.2f}/{:2.2f}, Loss(train/val): {:2.2f}/{:2.2f}. Lr: {:f}. Took {:0.2f} sec'.format(
                        epoch, train_acc, val_acc, train_auroc, val_auroc, train_loss, val_loss, optimizer.param_groups[0]["lr"], te - ts))

                early_stopping(val_acc, net, epoch, optimizer, fold = int(fold))

                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                if scheduler is not None:
                    writer_val.add_scalar('lr',optimizer.param_groups[0]["lr"], epoch)
                    lrs.append(optimizer.param_groups[0]["lr"])

                    if args.lr_policy == "plateau":
                        scheduler.step(val_acc)
                    else:
                        scheduler.step()

                # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
                # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            writer_train.close()
            writer_val.close()

            result['train_losses'] = train_losses
            result['train_accs'] = train_accs
            result['val_losses'] = val_losses
            result['val_accs'] = val_accs
            result['lr'] = lrs

        net, optim, st_epoch = load(ckpt_dir=args.ckpt_dir, net=net, optim=optimizer)
        test_acc, test_auc = test(net,partition,criterion,args)

        result['test_acc'] = test_acc
        result['test_auc'] = test_auc
        
        with open(args.result_dir + 'kfold_results.csv', 'a',encoding='utf-8',newline='') as f:
            wr = csv.writer(f)
            wr.writerow([fold, test_acc, test_auc])
    
    kfold_results = pd.read_csv(args.result_dir + 'kfold_results.csv')
    means = kfold_results.mean(axis=0)
    std = kfold_results.std(axis=0)
    kfold_results = kfold_results.append(means, ignore_index=True)
    new_df = kfold_results.append(std, ignore_index=True)
    new_df.to_csv(args.result_dir + 'kfold_results.csv')
    return vars(args), result

