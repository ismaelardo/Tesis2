import pandas as pd
import csv
from scipy.stats.stats import pearsonr
from os import listdir, mkdir
import numpy as np
# from IPython.display import clear_output
import pickle
import math
import torch
import time
import datetime
import torch.nn.utils.rnn as rnn
import torch.nn as nn
import torch.multiprocessing as mp
import random as rd
from torch.utils.data import Dataset, DataLoader
import copy
import torch.multiprocessing as mp
from torch.autograd import Variable

p_sql = 'data/sqls/data.sqlite3'
f_data = 'data/diccionarios/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_obj(folder, name):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, folder, name):
    with open(folder + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    # (A)crear data de features alterada, con ruido y eliminando ciertos valores del vector:(despues de normalizar)


features_crossval_standarized = load_obj(f_data, 'D_features_crossval_standarized_per_fold')
features_test_standarized = load_obj(f_data, 'D_features_test_standarized')
features_crossval_corto_standarized = load_obj(f_data, 'S_features_crossval_corto_standarized_per_fold')
valence_crossval_corto_standarized = load_obj(f_data, 'S_valence_crossval_corto_standarized_per_fold')
arousal_crossval_corto_standarized = load_obj(f_data, 'S_arousal_crossval_corto_standarized_per_fold')
features_test_corto_standarized = load_obj(f_data, 'S_features_test_corto_standarized')
valence_test_corto_standarized = load_obj(f_data, 'S_valence_test_corto_standarized')
arousal_test_corto_standarized = load_obj(f_data, 'S_arousal_test_corto_standarized')


def agregar_ruido(song, sd):
    ruido = np.random.normal(0, sd, song.shape)
    # print(type(song),type())
    feat_noised = song.float() + torch.from_numpy(ruido).float()
    return feat_noised


def destruir_caract(ejemplo, n):
    elim = rd.sample(range(260), n)
    for dele in elim:
        ejemplo[dele] = 0
    return ejemplo


# aplica agregar ruido con media cero y desviacion standar sd en cada song del batch
# (agregar_ruido(song,sd))y elimina n características en cada instante
# (destruir_caract(instant_feature,n))
# batch is a list
def corromper_batch(batch, sd, n):
    batch_feat_song_corrupted = []
    for feat_song in batch:
        batch_feat_song_corrupted.append(agregar_ruido(feat_song, sd))
        for t, instant_feat in enumerate(batch_feat_song_corrupted[-1]):
            batch_feat_song_corrupted[-1][t] = destruir_caract(batch_feat_song_corrupted[-1][t], n)
    return batch_feat_song_corrupted


# normalizar data
def normalizar_data(lista):
    listaNorm = []
    data = torch.cat(lista, dim=0)
    mean_lista = torch.mean(data, dim=0)
    std_lista = torch.std(data, dim=0)
    for feat_song in lista:
        listaNorm.append((feat_song - mean_lista) / std_lista)
    return listaNorm


# crear train data y validation data sin normalizar para cualquier batch de validación
def datatrain_dae(dicti, valid_batch, separar_train_folds=False):
    list_train = []
    list_valid = []
    if separar_train_folds:
        list_train_sep = []
    # data={}
    i = 0
    for batch in dicti:
        if batch != valid_batch:
            for song in dicti[batch]:
                list_train.append(dicti[batch][song])
            if separar_train_folds:
                list_train_sep.append(list_train[i:])
                i = len(list_train)
    for song in dicti[valid_batch]:
        list_valid.append(dicti[valid_batch][song])
    if separar_train_folds:
        # debuggeo
        # print(len(list_train_sep[0]))
        # imprime 40
        return list_train_sep, list_valid
    elif not separar_train_folds:
        return list_train, list_valid


def datatrain_super(dict_feature, dict_valence, dict_arousal, valid_batch, separar_train_folds=False):
    list_train_feature = []
    list_valid_feature = []
    list_train_valence = []
    list_valid_valence = []
    list_train_arousal = []
    list_valid_arousal = []
    if separar_train_folds:
        list_train_feature_sep = []
        list_train_valence_sep = []
        list_train_arousal_sep = []

    i = 0
    for batch in dict_feature:
        # print(batch,dict[batch].shape)
        if batch != valid_batch:
            for song in dict_feature[batch]:
                list_train_feature.append(dict_feature[batch][song])
                list_train_valence.append(dict_valence[batch][song].unsqueeze(1))
                list_train_arousal.append(dict_arousal[batch][song].unsqueeze(1))
            if separar_train_folds:
                list_train_feature_sep.append(list_train_feature[i:])
                list_train_valence_sep.append(list_train_valence[i:])
                list_train_arousal_sep.append(list_train_arousal[i:])
                i = len(list_train_feature)
    for song in dict_feature[valid_batch]:
        list_valid_feature.append(dict_feature[valid_batch][song])
        list_valid_valence.append(dict_valence[valid_batch][song].unsqueeze(1))
        list_valid_arousal.append(dict_arousal[valid_batch][song].unsqueeze(1))
    if separar_train_folds:
        return list_train_feature_sep, list_train_valence_sep, list_train_arousal_sep, list_valid_feature, list_valid_valence, list_valid_arousal
    elif not separar_train_folds:
        return list_train_feature, list_train_valence, list_train_arousal, list_valid_feature, list_valid_valence, list_valid_arousal


def datatrain_test_super(dict_feature, dict_valence, dict_arousal):
    list_test_feature = []
    list_test_valence = []
    list_test_arousal = []
    for song in dict_feature:
        list_test_feature.append(dict_feature[song])
        list_test_valence.append(dict_valence[song].unsqueeze(1))
        list_test_arousal.append(dict_arousal[song].unsqueeze(1))
    return list_test_feature, list_test_valence, list_test_arousal


def datatrain_test_dae(dicti):
    list_test = []
    for song in dicti:
        list_test.append(dicti[song])
    return list_test


'''
#transformar pad en tensor plano
def aplanar(padded,lengths):
  plana=torch.Tensor([]).to(device)
  for j in range(len(padded[0])):
    plana=torch.cat([plana,padded[:lengths[j],j]])
  return plana  
'''


# standarizar elementos de todos las song de cada fold a media y desviación estandar de su fold
def standarize_to_fold(dicti):
    sigma = []
    dicti_standar = copy.deepcopy(dicti)
    for fold in dicti:
        tensor_fold = torch.cat([dicti[fold][song] for song in dicti[fold]], dim=0)
        mean_fold = torch.mean(tensor_fold, dim=0)
        std_fold = torch.std(tensor_fold, dim=0)
        # print('mean_fold={},std_fold={}'.format(mean_fold[0],std_fold[0]))
        for song in dicti[fold]:
            std_song = torch.std(dicti[fold][song], dim=0)
            mean_song = torch.mean(dicti[fold][song], dim=0)
            dicti_standar[fold][song] = dicti[fold][
                                            song] * std_fold / std_song + mean_fold - mean_song * std_fold / std_song
            # print('mean_song={}, std_song={}'.format(torch.mean(dicti_standar[fold][song],dim=0)[0],torch.std(dicti_standar[fold][song],dim=0)[0]))
    return dicti_standar


def standarize_to_all_fold(dicti, valid_fold):
    sigma = []
    dicti_standar = copy.deepcopy(dicti)

    tensor_train = torch.cat([dicti[fold][song] for fold in dicti for song in dicti[fold] if fold != valid_fold], dim=0)
    # print(tensor_train.shape)
    mean_train = torch.mean(tensor_train, dim=0)
    std_train = torch.std(tensor_train, dim=0)
    # print('mean_train={},std_train={}'.format(mean_train[0],std_train[0]))
    for fold in dicti:
        if fold != valid_fold:
            for song in dicti[fold]:
                std_song = torch.std(dicti[fold][song], dim=0)
                mean_song = torch.mean(dicti[fold][song], dim=0)
                dicti_standar[fold][song] = dicti[fold][
                                                song] * std_train / std_song + mean_train - mean_song * std_train / std_song
                # print('mean_song={}, std_song={}'.format(torch.mean(dicti_standar[fold][song],dim=0)[0],torch.std(dicti_standar[fold][song],dim=0)[0]))
        else:
            tensor_valid_fold = torch.cat([dicti[fold][song] for song in dicti[fold]], dim=0)
            mean_valid = torch.mean(tensor_valid_fold, dim=0)
            std_valid = torch.std(tensor_valid_fold, dim=0)
            for song in dicti[fold]:
                std_song = torch.std(dicti[fold][song], dim=0)
                mean_song = torch.mean(dicti[fold][song], dim=0)
                dicti_standar[fold][song] = dicti[fold][
                                                song] * std_valid / std_song + mean_valid - mean_song * std_valid / std_song
                # print('mean_song={}, std_song={}'.format(torch.mean(dicti_standar[fold][song],dim=0)[0],torch.std(dicti_standar[fold][song],dim=0)[0]))
    return dicti_standar


# para estandarizar los test (que no estan separados por folds)
def standarize(dicti):
    sigma = []
    dicti_standar = copy.deepcopy(dicti)

    tensor_test = torch.cat([dicti[song] for song in dicti], dim=0)
    mean = torch.mean(tensor_test, dim=0)
    std = torch.std(tensor_test, dim=0)
    for song in dicti:
        std_song = torch.std(dicti[song], dim=0)
        mean_song = torch.mean(dicti[song], dim=0)
        dicti_standar[song] = dicti[song] * std / std_song + mean - mean_song * std / std_song
    return dicti_standar


# Dataloader y dataset
def my_collate_dae(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    return [data, label]


# Dataloader y dataset
def my_collate_super(batch):
    features = [item[0] for item in batch]
    valence = [item[1] for item in batch]
    arousal = [item[2] for item in batch]
    return [features, valence, arousal]


class featData_dae(Dataset):
    # Constructor
    def __init__(self, noiseFeatures, features):  # noiseFeatures y features son listas
        self.x = noiseFeatures
        self.y = features
        self.len = len(self.x)

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get length
    def __len__(self):
        return self.len


class featData_super(Dataset):
    # Constructor
    def __init__(self, list_features, list_valence,
                 list_arousal):  # noiseFeatures y features son diccionarios, no, son listas
        self.x = list_features
        self.y = list_valence
        self.z = list_arousal
        self.len = len(self.x)

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    # Get length
    def __len__(self):
        return self.len
    # Crear packed sequence de una lista de canciones (song)


def packInput(songs):
    vectorized_seqs = songs
    seq_lengths = torch.FloatTensor([len(seq) for seq in vectorized_seqs])  # .cuda() #58,....1223
    seq_tensor = torch.zeros(
        (len(vectorized_seqs), seq_lengths.long().max(), len(vectorized_seqs[0][0]))).float()  # .to(device)

    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths.long())):
        seq_tensor[idx, :seqlen, :] = torch.FloatTensor(seq.float())

    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # print(perm_idx,seq_lengths)
    seq_tensor = seq_tensor[perm_idx]

    # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
    # Otherwise, give (L,B,D) tensors
    seq_tensort = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)

    # embed your sequences
    # seq_tensor = embed(seq_tensor)

    # pack them up nicely
    packed_input = rnn.pack_padded_sequence(seq_tensort, seq_lengths.cpu().numpy())  # .to(device)
    return packed_input


def igualar_largos(features, valence, arousal):
    features2 = copy.deepcopy(features)
    valence2 = copy.deepcopy(valence)
    arousal2 = copy.deepcopy(arousal)
    for batch in features2:
        for song in features2[batch]:
            if features2[batch][song].shape[0] > valence2[batch][song].shape[0]:
                features2[batch][song] = features2[batch][song][:valence2[batch][song].shape[0], :]
                if features2[batch][song].shape[0] > arousal2[batch][song].shape[0]:
                    features2[batch][song] = features2[batch][song][:arousal2[batch][song].shape[0], :]
                else:
                    arousal2[batch][song] = arousal2[batch][song][:features2[batch][song].shape[0]]
            else:
                valence2[batch][song] = valence2[batch][song][:features2[batch][song].shape[0]]
                if valence2[batch][song].shape[0] > arousal2[batch][song].shape[0]:
                    valence2[batch][song] = valence2[batch][song][:arousal2[batch][song].shape[0], :]
                else:
                    arousal2[batch][song] = arousal2[batch][song][:valence2[batch][song].shape[0]]
    return features2, valence2, arousal2


def igualar_largos_test(features_test, valence_test, arousal_test):
    features_test2 = copy.deepcopy(features_test)
    valence_test2 = copy.deepcopy(valence_test)
    arousal_test2 = copy.deepcopy(arousal_test)
    for song in features_test2:
        if features_test2[song].shape[0] > valence_test2[song].shape[0]:
            features_test2[song] = features_test2[song][:valence_test2[song].shape[0], :]
            if features_test2[song].shape[0] > arousal_test2[song].shape[0]:
                features_test2[song] = features_test2[song][:arousal_test2[song].shape[0], :]
            else:
                arousal_test2[song] = arousal_test2[song][:features_test2[song].shape[0]]
        else:
            valence_test2[song] = valence_test2[song][:features_test2[song].shape[0]]
            if valence_test2[song].shape[0] > arousal_test2[song].shape[0]:
                valence_test2[song] = valence_test2[song][:arousal_test2[song].shape[0], :]
            else:
                arousal_test2[song] = arousal_test2[song][:valence_test2[song].shape[0]]

    return features_test2, valence_test2, arousal_test2


# Clase padre de modelos
class RNN(nn.Module):
    def __init__(self, parametros):
        super(RNN, self).__init__()

    def train(self):
        pass
    def test(self):
        pass
    def valid(self):
        pass

# (B)creación del modelo lstm-linear Este!pba padding linear
class DAE(RNN):
    def __init__(self, parametros):
        super(DAE, self).__init__(parametros)
        self.input_size = parametros['input_size_dae']
        self.output_size = parametros['output_size_dae']
        self.hidden_size = parametros['hidden_size_dae']
        self.LR = parametros['LR_dae']
        self.sd = parametros['sd_dae']
        self.batch_size = parametros['batch_size_dae']
        self.n = parametros['n_dae']
        self.criterion = parametros['criterion_dae']
        self.num_layers = 1
        self.h, self.c = self.init_hidden()
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)

    def change_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size

    def init_hidden(self):
        h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        return h, c

    def train(self, list_train_sep):
        for fold in list_train_sep:
            dataset = featData_dae(fold, fold.copy())
            trainloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=my_collate_dae,
                                     drop_last=True, shuffle=True)
            for x, y in trainloader:
                x = normalizar_data(x)
                x = corromper_batch(x, self.sd, self.n)
                y = normalizar_data(y)
                features_out = self(packInput(x).to(device))
                real_features = packInput(y).to(device)
                loss_train = self.criterion(features_out.data, real_features.data)
                self.optimizer.zero_grad()
                loss_train.backward()
                self.h, self.c = self.h.detach(), self.c.detach()
                self.optimizer.step()

    def valid(self, list_valid):
        list_loss_valid=[]
        with torch.no_grad():
            dataset = featData_dae(list_valid, list_valid.copy())
            trainloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                     collate_fn=my_collate_dae, drop_last=True, shuffle=True)
            for x, y in trainloader:
                x = normalizar_data(x)
                # x=corromper_batch(x,sd,n)
                y = normalizar_data(y)
                features_out = self(packInput(x).to(device))
                real_features = packInput(y).to(device)
                loss_valid = self.criterion(features_out.data, real_features.data)
                list_loss_valid.append(np.sqrt(loss_valid.item()))
        return list_loss_valid
    def test(self,list_test_feature):
        list_loss_test_valid=[]
        with torch.no_grad():
            dataset_test = featData_dae(list_test_feature, list_test_feature.copy())
            testloader = DataLoader(dataset=dataset_test, batch_size=self.batch_size,
                                    collate_fn=my_collate_dae, drop_last=False, shuffle=True)

            for features_list_input, feature_list_target in testloader:
                features_list_input = normalizar_data(features_list_input)
                feature_list_target = normalizar_data(feature_list_target)
                self.batch_size = len(features_list_input)
                self.h = self.h[:, :self.batch_size, :]
                self.c = self.c[:, :self.batch_size, :]
                features_out = self(packInput(features_list_input).to(device))
                real_features = packInput(feature_list_target).to(device)
                loss_test = self.criterion(features_out.data, real_features.data)
                list_loss_test_valid.append(np.sqrt(loss_test.item()))
        return np.mean(list_loss_test_valid)

    def forward(self, x):
        output, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        outpad = rnn.pad_packed_sequence(output)
        out = self.linear(outpad[0])
        out2 = rnn.pack_padded_sequence(out, outpad[1])
        return out2


# modelo clasificador
class LSTMsuper(RNN):
    def __init__(self,parametros, model):
        super(LSTMsuper, self).__init__(parametros)
        self.input_size = parametros['input_size_lstm']
        self.output_size = parametros['output_size_lstm']
        self.hidden_size = parametros['hidden_size_lstm']
        self.LR = parametros['LR_lstm']
        self.sd = parametros['sd_lstm']
        self.batch_size = parametros['batch_size_lstm']
        self.batch_size_test = parametros['batch_size_test_lstm']
        self.n = parametros['n_lstm']
        self.criterion = parametros['criterion_lstm']
        self.num_layers = 1
        self.hidden_size2 = parametros['hidden_size2_lstm']
        self.hidden_size3 = parametros['hidden_size3_lstm']
        self.h, self.c = self.init_hidden(self.hidden_size)
        self.h2, self.c2 = self.init_hidden(self.hidden_size2)
        self.h3, self.c3 = self.init_hidden(self.hidden_size3)
        self.dae = model
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size2)
        self.lstm3 = nn.LSTM(self.hidden_size2, self.hidden_size3)
        self.linear_valence = nn.Linear(self.hidden_size3, self.output_size)
        self.linear_arousal = nn.Linear(self.hidden_size3, self.output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)

    def init_hidden(self, hid_size):
        h = Variable(torch.zeros(self.num_layers, self.batch_size, hid_size)).to(device)
        c = Variable(torch.zeros(self.num_layers, self.batch_size, hid_size)).to(device)

        return h, c

    def change_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
    def train(self, list_train_feature_sep, list_train_valence_sep, list_train_arousal_sep):
        for foldindex in range(len(list_train_feature_sep)):
            # print(foldindex)
            dataset_train_lstm = featData_super(
                list_train_feature_sep[foldindex],
                list_train_valence_sep[foldindex],
                list_train_arousal_sep[foldindex])
            trainloader_lstm = DataLoader(
                dataset=dataset_train_lstm,
                batch_size=self.batch_size,
                collate_fn=my_collate_super,
                drop_last=True, shuffle=True)
            for features_list, valence_list, arousal_list in trainloader_lstm:
                features_list = normalizar_data(features_list)
                valence_list = normalizar_data(valence_list)
                arousal_list = normalizar_data(arousal_list)
                features_list = corromper_batch(features_list,
                                                self.sd, self.n)
                valence_pack = packInput(valence_list).to(
                    device)
                arousal_pack = packInput(arousal_list).to(
                    device)
                annotations_target = torch.cat(
                    [valence_pack.data, arousal_pack.data])
                valence_out, arousal_out = self(
                    packInput(features_list).to(device))
                annotations_out = torch.cat(
                    [valence_out.data, arousal_out.data])
                loss_train = self.criterion(annotations_out,
                                       annotations_target)
                self.optimizer.zero_grad()
                loss_train.backward()
                self.h = self.h.detach()
                self.c = self.c.detach()
                self.h2 = self.h2.detach()
                self.c2 = self.c2.detach()
                self.h3 = self.h3.detach()
                self.c3 = self.c3.detach()
                self.optimizer.step()

    def valid(self, list_valid_feature, list_valid_valence, list_valid_arousal, listLoss_valid=[],
              listLoss_valid_valence=[], listLoss_valid_arousal=[]):
        with torch.no_grad():
            dataset_valid_lstm = featData_super(
                list_valid_feature,
                list_valid_valence,
                list_valid_arousal)
            trainloader_valid_lstm = DataLoader(
                dataset=dataset_valid_lstm,
                batch_size=self.batch_size,
                collate_fn=my_collate_super,
                drop_last=True, shuffle=True)

            for features_list, valence_list, arousal_list in trainloader_valid_lstm:
                features_list = normalizar_data(features_list)
                valence_list = normalizar_data(valence_list)
                arousal_list = normalizar_data(arousal_list)
                valence_pack = packInput(valence_list).to(
                    device)
                arousal_pack = packInput(arousal_list).to(
                    device)
                annotations_target = torch.cat(
                    [valence_pack.data, arousal_pack.data])
                valence_out, arousal_out = self(
                    packInput(features_list).to(device))
                annotations_out = torch.cat(
                    [valence_out.data, arousal_out.data])
                loss_valid = self.criterion(annotations_out,
                                            annotations_target)
                loss_valid_valence = self.criterion(valence_out.data,
                                                    valence_pack.data)
                loss_valid_arousal = self.criterion(arousal_out.data,
                                                    arousal_pack.data)
                listLoss_valid.append(
                    np.sqrt(loss_valid.item()))
                listLoss_valid_valence.append(
                    np.sqrt(loss_valid_valence.item()))
                listLoss_valid_arousal.append(
                    np.sqrt(loss_valid_arousal.item()))

        return listLoss_valid,listLoss_valid_valence,listLoss_valid_arousal
    def test(self,features_test_corto_standarized,valence_test_corto_standarized,arousal_test_corto_standarized):
        listLoss_test = []
        listLoss_test_valence = []
        listLoss_test_arousal = []
        self.change_batch_size(self.batch_size_test)
        self.h, self.c = self.init_hidden(
            self.hidden_size)
        self.h2, self.c2 = self.init_hidden(
            self.hidden_size2)
        self.h3, self.c3 = self.init_hidden(
            self.hidden_size3)
        list_test_feature, list_test_valence, list_test_arousal = datatrain_test_super(
            features_test_corto_standarized,
            valence_test_corto_standarized,
            arousal_test_corto_standarized)
        with torch.no_grad():
            dataset_test_lstm = featData_super(list_test_feature,
                                               list_test_valence,
                                               list_test_arousal)
            testloader_test_lstm = DataLoader(dataset=dataset_test_lstm,
                                              batch_size=self.batch_size_test,
                                              collate_fn=my_collate_super,
                                              drop_last=True,
                                              shuffle=True)

            for features_list, valence_list, arousal_list in testloader_test_lstm:
                features_list = normalizar_data(features_list)
                valence_list = normalizar_data(valence_list)
                arousal_list = normalizar_data(arousal_list)
                valence_pack = packInput(valence_list).to(device)
                arousal_pack = packInput(arousal_list).to(device)
                annotations_target = torch.cat(
                    [valence_pack.data, arousal_pack.data])
                valence_out, arousal_out = self(
                    packInput(features_list).to(device))
                annotations_out = torch.cat(
                    [valence_out.data, arousal_out.data])
                loss_test = self.criterion(annotations_out,
                                           annotations_target)
                loss_test_valence = self.criterion(valence_out.data,
                                                   valence_pack.data)
                loss_test_arousal = self.criterion(arousal_out.data,
                                                   arousal_pack.data)
                listLoss_test.append(loss_test.item())
                listLoss_test_valence.append(
                    np.sqrt(loss_test_valence.item()))
                listLoss_test_arousal.append(
                    np.sqrt(loss_test_arousal.item()))
        return listLoss_test, listLoss_test_valence, listLoss_test_arousal
    def forward(self, x):
        output, (self.h, self.c) = self.dae(x, (self.h, self.c))
        # print(output.data.shape)
        output2, (self.h2, self.h2) = self.lstm2(output, (self.h2, self.h2))
        output3, (self.h3, self.h3) = self.lstm3(output2, (self.h3, self.h3))
        # print(output.data.shape)
        outpad = rnn.pad_packed_sequence(output3)
        out_valence = self.linear_valence(outpad[0])
        out_arousal = self.linear_arousal(outpad[0])
        out2_valence = rnn.pack_padded_sequence(out_valence, outpad[1])
        out2_arousal = rnn.pack_padded_sequence(out_arousal, outpad[1])
        return out2_valence, out2_arousal
