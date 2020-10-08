import numpy as np
import sys
# sys.path.append('D:\PythonDemo\PytorchDemo\mini_EEGNet\EEGData')
import scipy.io as sio
import datetime
import os
from EEGData.EEGDataset import *
import h5py
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from Models import *

class TrainModel():
    def __init__(self):
        self.data = None
        self.label = None
        self.result = None
        self.input_shape = None  # should be (width, height)
        self.EEGNet_inputsize = None
        self.LSTM_inputsize = None
        self.model = 'EEGNet'
        self.cross_validation = 'Session'
        self.ampling_rate = 128
        self.file_dir = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Parameters: Training process
        self.random_seed = 42
        self.learning_rate = 1e-3
        self.num_epochs = 300
        self.num_classes = 2
        self.batch_size = 23
        self.patient = 4
        self.K = 5
        self.valence_or_arousal = None

        # Parameters: Model
        self.dropout_rate = 0.3
        self.F1 = 5
        self.D = 2
        self.F2 = 16
        self.hidden_node = 64
        self.Lambda = 1e-6
        self.input_channel = 5

    # def load_data(self, file_dir, file, subject_dependent, flag):
    #     files = os.listdir(file_dir)
    #     if subject_dependent.__eq__(True): # 实验是否为subject_dependent，为true，则导入file参数指明的文件
    #         mat = sio.loadmat(file_dir + file)
    #         data = mat['Data']
    #         if flag == 'valence':
    #             labels = mat['valence_labels']
    #         elif flag == 'arousal':
    #             labels = mat['arousal_labels']
    #         elif flag == 'four_classes':
    #             labels = np.array(mat['labels'])
    #         else:
    #             print("flag 参数错误，应该为'valence'或'arousal'")
    #     elif subject_dependent.__eq__(False):
    #         mat = sio.loadmat(file_dir + files[0])
    #         data = mat['Data']
    #         if flag == 'valence':
    #             labels = mat['valence_labels']
    #             labels = np.array(labels).T
    #         elif flag == 'arousal':
    #             labels = mat['arousal_labels']
    #             labels = np.array(labels).T
    #         elif flag == 'four_classes':
    #             labels = mat['labels']
    #             labels = np.array(labels).T
    #         else:
    #             print("flag 参数错误，应该为'valence'或'arousal'")
    #         for i in range(1, len(files)):
    #             file_path = file_dir + files[i]
    #             mat = sio.loadmat(file_path)
    #             data = np.vstack((data, mat['Data']))
    #             #             type1 = mat['Types']
    #             #             type1  = np.reshape(type1, (240,1))
    #             #             types = np.vstack((types,type1))
    #             if flag == 'valence':
    #                 labels = np.vstack((labels, np.array(mat['valence_labels']).T))
    #             elif flag == 'arousal':
    #                 labels = np.vstack((labels, np.array(mat['arousal_labels']).T))
    #             elif flag == 'four_classes':
    #                 labels = np.vstack((labels, np.array(mat['labels']).T))
    #             else:
    #                 print("flag 参数错误，应该为'valence'或'arousal'")
    #     else:
    #         print("参数错误，请检查！")
    #     labels = np.squeeze(labels)
    #     self.data = data
    #     self.label = labels
    #     eegnetdata = data[:,0:5,:,:]
    #     lstmdata = data[:,-1,:,:]
    #     self.input_shape = self.data.shape
    #     self.EEGNet_inputsize = eegnetdata.shape
    #     self.LSTM_inputsize = lstmdata.shape
    #
    #     print('Data loaded!\n Data shape:[{}], Label shape:[{}]'.format(self.data.shape, self.label.shape))

    def load_data(self, path, valence_or_arousal):
        files = os.listdir(path)
        for i in range(len(files)):
            file_path = path + files[i]
            dataset = sio.loadmat(file_path)
            if i == 0:
                self.data = np.array(dataset['Data'])
                # self.data = self.data.reshape((1, self.data.shape[0, self.data.shape[1], 1, self.data.shape[2], self.data.shape[3]))
                self.data = self.data.reshape(
                    (1, self.data.shape[0], self.data.shape[1], self.data.shape[2], self.data.shape[3], self.data.shape[4]))
                if valence_or_arousal.__eq__('valence'):
                    self.label = np.array(dataset['valence_labels'])
                elif valence_or_arousal.__eq__('arousal'):
                    self.label = np.array(dataset['arousal_labels'])
                self.label = self.label.reshape(1, 40, 23)
            else:
                data = np.array(dataset['Data'])
                data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2], data.shape[3], data.shape[4])
                if valence_or_arousal.__eq__('valence'):
                    label = np.array(dataset['valence_labels'])
                elif valence_or_arousal.__eq__('arousal'):
                    label = np.array(dataset['arousal_labels'])
                label = label.reshape(1, 40, 23)
                self.data = np.concatenate((self.data, data), axis=0)
                self.label = np.concatenate((self.label, label), axis=0)
        # The input_shape should be (channel x data)
        self.input_shape = self.data[0, 0, 0].shape

        print('Data loaded!\n Data shape:[{}], Label shape:[{}]'
              .format(self.data.shape, self.label.shape))

    def set_parameter(self, cv, model, number_class, random_seed, learning_rate, epoch,
                      batch_size, dropout_rate, F1, D, F2, hidden_node, patient, Lambda,
                      input_channel, file_dir, K, valence_or_arousal):
        '''
        This  is the function to set the parameters of tarining process and model
        ALl the settings wil be saved into a NAME.txt file
        Input:  cv--
                    The cross-validation type
                    Type=String
                    Default: Leave_out_session_out
                    Note: For different cross validation type, please add the
                      corresponding cross validation function.(e.g. self.leave_one_session_out())


                model: --
                    The model you want choose
                    Type = String
                    Default: mini_EEGNet
                number_class: --
                    The number  of classes
                    Type = int
                    Default: 2
                random_seed: --
                     The random seed
                     Type: int
                     Default: 42
                learning_rate: --
                    Learning rate
                    Type: float
                    Default: 0.0001
                epoch: --
                    Type: int
                    Default: 300
                batch_size: --
                    The size of mini-batch
                    Type: int
                    Default: 24
                dropout_rate: --
                    dropout rate of the fully Conv2d layers
                    Type: float
                    Default: 0.3
                F1: --
                    The channel number of the input of block1.
                    Type: int
                    Default: 5
                D: --
                    The number of spatial filter of block1
                    Type: int
                    Default: 2
                F2: --
                    The number of the channel of point-Conv2d
                    Type: int
                    Default: 16
                hidden_node: --
                    The hidden node of lstm block
                    Type: int
                    Default: 64
                patient: --
                    How many epoches the training process hsould wait for
                    It is used for the early-stopping
                    Type: int
                    Default: 4
                 Lambda --
                   The L1 regulation coefficient in loss function
                   Type : float
                   Default : 1e-6
                Input_channel： --
                    The channel of the input of model.
                    Type: int
                    Default: 5
        '''
        self.model = model
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = epoch
        self.num_classes = number_class
        self.batch_size = batch_size
        self.patient = patient
        self.Lambda = Lambda
        self.K = K
        self.valence_or_arousal = valence_or_arousal

        # Parameters: Model
        self.dropout_rate = dropout_rate
        self.hidden_node = hidden_node
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.input_channel = input_channel
        self.file_dir = file_dir

        # Save to log file for checking
        if cv =="Leave_one_subject_out":
            file = open("result_subject.txt", 'a')
        elif cv =="Leave_one_session_out":
            file = open("result_session.txt", "a")
        elif cv == "k_fold_validation":
            file = open("result_k_fold.txt", "a")
        file.write("\n"+str(datetime.datetime.now())+
                   "\nTrain:Parameter setting for"+str(self.model)+
                   "\n1)number class:"+str(self.num_classes) + "\n2)random_seed:"+str(self.random_seed)+
                   "\n3)learning_rate:"+str(self.learning_rate) + "\n4)num_epochs:"+str(self.num_epochs)+
                   "\n5)batch_size:"+str(self.batch_size)+
                   "\n6)dropout_rate:"+str(self.dropout_rate)+"\n7)F1:"+str(self.F1)+
                   "\n8)D:"+str(self.D) + "\n9)input_shape:"+str(self.input_shape)+
                   "\n10)input_channel:"+str(self.input_channel)+"\n11)patient:"+str(self.patient)+
                   "\n12)Lambda:"+str(self.Lambda)+'\n13)file_dir:' + str(self.file_dir))
        file.close()

    # def k_fold_validation(self):
    #     '''
    #     This is the function to achieve 'k fold' cross-validation
    #     Note : all the acc and std will be logged into the result_session.txt
    #            The txt file is located at the same location as the python script
    #     '''
    #     save_path = Path(os.getcwd())
    #     if not os.path.exists(save_path / Path('Result_model/K_fold_validation/history')):
    #         os.makedirs(save_path/Path('Result_model/k_fold_validation/history'))
    #
    #     # Train and evaluate the model subject by subject
    #     ACC = []
    #     ACC_mean = []
    #     files = os.listdir(self.file_dir)
    #     for i in range(len(files)):
    #         self.load_data(self.file_dir, files[i], True, self.valence_or_arousal)
    #
    #         # Data dimension:  segments x channels x height x width
    #         # Label dimension:  segments
    #         data = self.data
    #         label = self.label
    #         shape_data = data.shape
    #         shape_label = label.shape
    #         channel = shape_data[1]
    #         height = shape_data[2]
    #         width = shape_data[3]
    #         print("Train:K_flod_validation \n1)shape of data:" + str(shape_data) + "\n2)shape of label:" + str(
    #             shape_label) +
    #               " \n3)channel:" + str(channel) + "\n4)height:" + str(height) + "\n5)width:" + str(width))
    #
    #         ACC_subject = []
    #         ACC_fold = []
    #         for curr_fold in range(self.K):
    #
    #             fold_size = self.data.shape[0] // self.K
    #             indexes_list = [i for i in range(len(self.data))]
    #             indexes = np.array(indexes_list)
    #             split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
    #             split = np.array(split_list)
    #             data_test  = self.data[split]
    #             label_test = self.label[split]
    #             split = np.array(list(set(indexes_list) ^ set(split_list)))
    #             data_train  = self.data[split]
    #             label_train = self.label[split]
    #
    #             # Split the training set into training set and validation set
    #             data_train, label_train, data_val, label_val = self.split(data_train, label_train)
    #
    #             # Prepare the data format for training the model
    #             data_train = torch.from_numpy(data_train).float()
    #             label_train = torch.from_numpy(label_train).long()
    #
    #             data_val = torch.from_numpy(data_val).float()
    #             label_val = torch.from_numpy(label_val).long()
    #
    #             data_test = torch.from_numpy(data_test).float()
    #             label_test = torch.from_numpy(label_test).long()
    #
    #             # Check the dimension of the training, validation and test set
    #             print('Training:', data_train.size(), label_train.size())
    #             print('Validation:', data_val.size(), label_val.size())
    #             print('Test:', data_test.size(), label_test.size())
    #
    #             # Get the accuracy of the model
    #             ACC_fold = self.train(data_train, label_train, data_test, label_test,
    #                                   data_val, label_val, subject=files[i], fold=curr_fold, cv_type="k_fold_validation")
    #             ACC_subject.append(ACC_fold)
    #             '''
    #             # Log the result per session
    #             file = open("result_fold.txt", 'a')
    #             file.write('Subject:' + str(files[i]) + 'Fold:'+str(curr_fold) + 'ACC:'+ str(ACC_session) + '\n')
    #             file.close()
    #             '''
    #         ACC_subject = np.array(ACC_subject)
    #         mAcc = np.mean(ACC_subject)
    #         std = np.std(ACC_subject)
    #
    #         print("Subject:" + str(files[i]) + "\nmACC: %.2f"% mAcc)
    #         print("std:%.2f" % std)
    #
    #         # Log the result per subject
    #         file = open("result_fold.txt", 'a')
    #         file.write('Model:'+ str(self.model) +' Subject:' + str(files[i]) + ' MeanACC:'+str(mAcc) + " Std:" + str(std) + '\n')
    #         file.close()
    #
    #         ACC.append(ACC_subject)
    #         ACC_mean.append(mAcc)
    #
    #     self.result = ACC
    #     # Log the final Acc and std of all the subjects
    #     file = open("result_fold.txt", 'a')
    #     file.write("\n" + str(datetime.datetime.now()) + "\nMeanACC:" + str(np.mean(ACC_mean)) + " Std:" + str(np.std(ACC_mean)) + '\n')
    #     file.close()
    #     print("Mean ACC:" + str(np.mean(ACC_mean)) + 'std:' + str(np.std(ACC_mean)))
    #
    #     # Save the model
    #     save_path = Path(os.getcwd())
    #     filename_data = save_path / Path("Result_mode/Result.hdf")
    #     save_data = h5py.File(filename_data, 'w')
    #     save_data['result'] = self.result
    #     save_data.close()

    def k_fold_validation(self):
        '''
        This is the function to achieve 'k fold' cross-validation
        Note : all the acc and std will be logged into the result_session.txt
               The txt file is located at the same location as the python script
        '''
        save_path = Path(os.getcwd())
        # if not os.path.exists(save_path / Path('Result_model/K_fold_validation/history')):
        #     os.makedirs(save_path/Path('Result_model/k_fold_validation/history'))
        # Data dimension: subject x trials x segments x band x psi_width x psi_height
        # Label dimension: subject x trials x segments
        # Session: trials[0:2]-session 1; trials[2:4]-session 2; trials[4:end]-session 3
        data  = self.data
        label = self.label
        shape_data = data.shape
        shape_label = label.shape
        subject = shape_data[0]
        trials = shape_data[1]


        channel = shape_data[3]
        print("")


        # Train and evaluate the model subject by subject
        ACC = []
        ACC_mean = []
        for i in range(subject):
            index = np.arange(trials)
            fold_size = trials // self.K
            ACC_subject = []
            ACC_fold = []
            for j in range(self.K):
                # Split the data into training set and test set
                # One fold(contains 8 trials) is test set
                # The rest are training set
                split = np.arange(j*fold_size, (j+1)*fold_size)
                index_train = np.delete(index, split)
                index_test = index[split]

                data_train = data[i, index_train, :, :, :, :]
                label_train = label[i, index_train, :]

                data_test = data[i, index_test, :, :, :, :]
                label_test = label[i, index_test, :]

                # Split the training set into training set and validation set
                data_train, label_train, data_val, label_val = self.split(data_train, label_train)

                # Prepare the data format for training the model
                data_train = torch.from_numpy(data_train).float()
                label_train = torch.from_numpy(label_train).long()

                data_val = torch.from_numpy(data_val).float()
                label_val = torch.from_numpy(label_val).long()

                data_test = torch.from_numpy(np.concatenate(data_test, axis=0)).float()
                label_test = torch.from_numpy(np.concatenate(label_test, axis=0)).long()

                # Check the dimension of the training, validation and test set
                print("Training:", data_train.size(), label_train.size())
                print("Validation:", data_val.size(), label_val.size())
                print("Test:", data_test.size(), label_test.size())

                # Get the accuracy of the model
                ACC_fold = self.train(data_train, label_train,
                                      data_test, label_test,
                                      data_val, label_val,
                                      subject=i, fold=j,
                                      cv_type="k_fold_validiton")

                ACC_subject.append(ACC_fold)
                '''
                Log the results per session
                。。。
                '''
            ACC_subject = np.array(ACC_subject)
            mAcc = np.mean(ACC_subject)
            std = np.std(ACC_subject)

            print("Subject:" + str(i) + "\nmACC: %.2f" % mAcc)
            print("std: %.2f" % std)

            # Log the results per subject
            # Log the results per subject
            file = open("result_k_fold.txt", 'a')
            file.write('Subject:' + str(i) + ' MeanACC:' + str(mAcc) + ' Std:' + str(std) + '\n')
            file.close()

            ACC.append(ACC_subject)
            ACC_mean.append(mAcc)

        self.result = ACC
        # Log the final Acc and std of all the subjects
        file = open("result_fold.txt", 'a')
        file.write("\n" + str(datetime.datetime.now()) + "\nMeanACC:" + str(np.mean(ACC_mean)) + " Std:" + str(np.std(ACC_mean)) + '\n')
        file.close()
        print("Mean ACC:" + str(np.mean(ACC_mean)) + 'std:' + str(np.std(ACC_mean)))

        # Save the model
        save_path = Path(os.getcwd())
        filename_data = save_path / Path("Result_mode/Result.hdf")
        save_data = h5py.File(filename_data, 'w')
        save_data['result'] = self.result
        save_data.close()


    def split(self, data, label):
        '''
        This is the function to split the training set into training set and validation set
        Input : data --
                The training data
                Dimension : trials x segments x 1 x channel x data
                Type : np.array

                label --
                The label of training data
                Dimension : trials x segments
                Type : np.array

        Output : train --
                 The split training data
                 Dimension : trials x segments x 1 x channel x data
                 Type : np.array

                 train_label --
                 The corresponding label of split training data
                 Dimension : trials x segments
                 Type : np.array

                 val --
                 The split validation data
                 Dimension : trials x segments x 1 x channel x data
                 Type : np.array

                 val_label --
                 The corresponding label of split validation data
                 Dimension : trials x segments
                 Type : np.array
        '''
        # Data dimension: trials x segments x 1 x channel x data
        # Label dimension: trials x segments
        np.random.seed(0)
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        # data : segments x 1 x channel x data
        # label : segments

        index = np.arange(data.shape[0])
        index_random = index

        np.random.shuffle(index_random)
        data = data[index_random]
        labels = label[index_random]

        # get validation set
        val_data = data[int(data.shape[0]*0.8):]
        val_labels = labels[int(data.shape[0]*0.8):]

        # get train set
        train_data = data[0:int(data.shape[0]*0.8)]
        train_labels = label[0:int(data.shape[0]*0.8)]

        return train_data, train_labels, val_data, val_labels

    def make_train_step(self, model, loss_fn, optimzier):
        def train_step(x, y):
            model.train()
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item()/len(pred)
            # L1 regularization
            loss_r = self.regulization(model, self.Lambda)
            # yhat is one-hot representation;
            # loss = loss_fn(yhat, y) + loss_r
            loss = loss_fn(yhat, y)
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            return loss.item(), acc
        return train_step

    def regulization(self, model, Lambda):
        w = torch.cat([x.view(-1) for x in model.parameters()])
        err = Lambda * torch.sum(torch.abs(w))
        return err

    def train(self, train_data, train_label, test_data, test_label, val_data,
              val_label, subject, fold, cv_type):
        print("Available Device:"+str(torch.cuda.get_device_name(torch.cuda.current_device())))
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        # Train and Validation loss
        losses = []
        accs = []

        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []

        test_losses = []
        test_acc = []
        Acc_test = []

        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs

        # build the model
        if self.model == 'mini_EEGNet':
            model= mini_EEGNet(num_classes=self.num_classes, input_size=self.input_shape,
                               input_channel=self.input_channel, F1=self.F1, D=self.D, F2=self.F2,
                               dropout_rate=self.dropout_rate)
        elif self.model == "LSTM_Net":
            model = LSTM_Net(num_classes=self.num_classes)
        elif self.model == "EEGNet_LSTM":
            model = EEGNet_LSTM(num_classes=self.num_classes, inputsize=self.input_shape,
                                F1=self.F1, D=self.D, F2=self.F2,
                                dropout_rate=self.dropout_rate, input_channel=self.input_channel)

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step(model, loss_fn, optimizer)

        # load the data
        dataset_train = EEGDataset(train_data, train_label)
        dataset_val = EEGDataset(val_data, val_label)
        dataset_test = EEGDataset(test_data, test_label)

        # Dataloader for training process
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(dataset=dataset_val, batch_size=self.batch_size,shuffle=True, pin_memory=False)
        test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=True, pin_memory=False)
        total_step = len(train_loader)

        ################Training process######################
        Acc = []
        acc_max = 0
        patient = 0

        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            for i, (x_batch, y_batch) in enumerate(train_loader):

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)

            losses.append(sum(loss_epoch)/len(loss_epoch))
            accs.append(sum(acc_epoch)/len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print('Epoch [{}/{}], Loss:{:.4f}, Acc:{:.4f}'.format(epoch+1, num_epochs, losses[-1], accs[-1]))

            ##############Validation process####################
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    model.eval()

                    yhat = model(x_val)
                    pred = yhat.max(1)[1]
                    # print("------")
                    # print(pred.size())
                    # print(y_val.size())
                    # print("------")
                    correct = (pred == y_val).sum()
                    acc = correct.item()/len(pred)
                    val_loss = loss_fn(yhat, y_val)
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)

                Acc_val.append(sum(val_acc)/len(val_acc))
                Loss_val.append(sum(val_losses)/len(val_losses))
                print('Evaluation Loss:{:.4f}, Acc:{:.4f}'.format(Loss_val[-1], Acc_val[-1]))
                val_losses = []
                val_acc = []

            ############early stop###################3
            Acc_es = Acc_val[-1]

            if Acc_es > acc_max:
                acc_max = Acc_es
                patient = 0
                print('------Model Saved!-------')
                torch.save(model, 'valence_max_model.pt')
            else:
                patient += 1
            if patient > self.patient:
                print('-----Early stopping-------')
                break

        #########test process############
        model = torch.load('valence_max_model.pt')
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                model.eval()

                yhat = model(x_test)
                pred = yhat.max(1)[1]
                correct = (pred == y_test).sum()
                acc = correct.item()/len(pred)
                test_loss = loss_fn(yhat, y_test)
                test_losses.append(test_loss.item())
                test_acc.append(acc)

            print('Test Loss:{:.4f}, Acc:{:.4f}'
                  .format(sum(test_losses)/len(test_losses), sum(test_acc)/len(test_acc)))
            Acc_test = (sum(test_acc)/len(test_acc))
            test_losses = []
            test_acc = []
        # save the loss(acc) for plotting the loss(acc) curve
        save_path = Path(os.getcwd())
        if cv_type == "k_fold_validation":
            filename_callback = save_path / Path('Result_model/k_fold_validation/history/'
                                                 +'history_subject_' + str(subject) + '_session_'
                                                 +str(fold) + '_history.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['val_acc'] = Acc_val
            save_history['loss'] = losses
            save_history['val_loss'] = Loss_val
            save_history.close()
        return Acc_test


if __name__ == '__main__':
    train = TrainModel()
    train.load_data('/home/whb/DataSets/intergrated_5s_window_deap_psi/', 'arousal')
    train.set_parameter(cv="k_fold_validation",
                        model='mini_EEGNet',
                        number_class=2,
                        random_seed=42,
                        learning_rate=1e-3,
                        epoch=300,
                        batch_size=11,
                        hidden_node=11,
                        patient=40,
                        Lambda=1e-6,
                        input_channel=5,
                        file_dir="/home/whb/DataSets/intergrated_processed_data/",
                        #                         file_dir="D:\\Matlab\\Matlab-Learn-master\\newnew\\intergrated_processed_data\\",
                        K=5,
                        valence_or_arousal="valence",
                        dropout_rate=0.5,
                        F1=5,
                        D=2,
                        F2=16,
                        )
    train.k_fold_validation()
