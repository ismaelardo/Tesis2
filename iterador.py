from entity import *
from guardar import DatosSQL

# lineas que se guardaran como comentario
l1 = '#parametros de estructura dae'
l2 = 'dae_init_h_state_per="epoch"# esto puede ser "epoch", "val_fold", "fold" o "batch"'
l3 = 'dae_standarization_per="each_fold" # puede ser "all_train_folds" o "each_fold"'
l4 = 'dae_datatrain_entregada="per_fold" # puede ser "completa" o "per_fold"'
l5 = 'dae_optim_per="batch"'
l6 = 'dae_normalizar_corromper=True'
l7 = '#parametros de estructura lstmsuper'
l8 = 'init_h_state_per="epoch"# esto puede ser "epoch", "val_fold", "fold" o "batch"'
l9 = 'standarization_per="each_fold" # puede ser "all_train_folds" o "each_fold"'
l10 = 'datatrain_entregada="per_fold" # puede ser "completa" o "per_fold"'
l11 = 'optim_per="batch"'
commentary = l1 + '\n' + l2 + '\n' + l3 + '\n' + l4 + '\n' + l5 + '\n' + l6 + '\n' + l7 + '\n' + l8 + '\n' + l9 + '\n' + l10 + '\n' + l11

req_grad_dae = 0
output_size_lstm = 1
num_epoch = 10
input_size = 260
output_size_dae = 260
num_layers = 1
batch_size_test = 10
pba = 0
criterion = nn.MSELoss(reduction='mean')
for sd_dae in [0.2, 0.3, 0.4]:
    for learning_rate_dae in [5e-6, 5e-4]:

        for hidden_size_dae in [50, 200, 800]:
            for batch_size_dae in [5, 10, 20]:
                for n_dae in [0, 10, 50]:
                    parametros = {'sd': sd_dae, 'LR': learning_rate_dae, 'hidden_size': hidden_size_dae,
                                  'batch_size': batch_size_dae,
                                  'n': n_dae, 'num_epoch': num_epoch, 'criterion': criterion}
                    dae = DAE(input_size, num_layers, output_size_dae, parametros).to(
                        device)
                    # dae.load_state_dict(torch.load(f_estructura+'Dae_loss_test_mean=0.457_loss_test_sd=0.033/'+'ModeloDAE'))
                    sobre_entrenamiento = False

                    for it_dae in range(10):
                        pba = pba + 1
                        start_time = time.time()

                        loss_valid_mean, loss_train_mean, listLoss_valid, listLoss_train, listBest_loss_valid_mean, i_best_loss_valid_mean = dae.train()

                        # print(len(listLoss_valid_epoch),len(listLoss_train_epoch))

                        elapsed_time_train = datetime.timedelta(seconds=time.time() - start_time)

                        # Testeo del modelo
                        listLoss_test = []
                        dae.batch_size = batch_size_test
                        dae.h, dae.c = dae.init_hidden()

                        # lstmsuper.dae.change_batch_size(batch_size_test)
                        list_test_feature = datatrain_test_dae(features_test_standarized)
                        with torch.no_grad():
                            dataset_test = featData_dae(list_test_feature, list_test_feature.copy())
                            testloader = DataLoader(dataset=dataset_test, batch_size=batch_size_test,
                                                    collate_fn=my_collate_dae, drop_last=False, shuffle=True)

                            for features_list_input, feature_list_target in testloader:
                                features_list_input = normalizar_data(features_list_input)
                                feature_list_target = normalizar_data(feature_list_target)
                                dae.batch_size = len(features_list_input)
                                dae.h = dae.h[:, :dae.batch_size, :]
                                dae.c = dae.c[:, :dae.batch_size, :]
                                features_out = dae(packInput(features_list_input).to(device))
                                real_features = packInput(feature_list_target).to(device)
                                loss_test = criterion(features_out.data, real_features.data)
                                listLoss_test.append(np.sqrt(loss_test.item()))

                        loss_test_std = np.std(listLoss_test)
                        loss_test_mean = np.mean(listLoss_test)

                        dae.batch_size = batch_size_dae
                        elapsed_time = datetime.timedelta(seconds=time.time() - start_time)

                        schema_infodae = {
                            'Loss_test_mean': loss_test_mean,
                            'Loss_test_sd': loss_test_std,

                            'Loss_valid_mean': loss_valid_mean,
                            'Loss_train_mean': loss_train_mean,
                            'n': n_dae,
                            'epochs': num_epoch,
                            'sd': sd_dae,
                            'batch_size': batch_size_dae,
                            'batch_size_test': batch_size_test,
                            'hidden_size': hidden_size_dae,
                            'LR': learning_rate_dae,
                            'tiempo_train': str(elapsed_time_train),
                            'tiempo_total': str(elapsed_time),
                            'sobre_entrenamiento': sobre_entrenamiento,
                            'estructura': 1,  # DONDE SACO ESTE DATO?,
                            'evol_loss_valid': pickle.dumps(listLoss_valid),
                            'evol_loss_train': pickle.dumps(listLoss_train),
                            'evol_loss_test': pickle.dumps(listLoss_test),
                            'list_best_loss_valid_mean': pickle.dumps(listBest_loss_valid_mean),
                            'index_best_loss_valid_mean': pickle.dumps(i_best_loss_valid_mean),
                            'ModeloDAE': pickle.dumps(dae.state_dict()),
                            'pba': pba,
                            'it': it_dae
                            # 'comentario': commentarydae
                        }
                        data = DatosSQL(p_sql)
                        data.guardar(schema_infodae)

                        sobre_entrenamiento = f'Loss_test_mean={loss_test_mean}, Loss_test_sd={loss_test_std}'
                        print(
                            'sd={},learing_rate={},hidden_size={},batch_size={},n={},it={},lossTestMean={},lossTestSTD={},Tiempo transcurrido={}'.format(
                                sd_dae, learning_rate_dae, hidden_size_dae, batch_size_dae, n_dae, it_dae,
                                loss_test_mean,
                                loss_test_std, elapsed_time))
                        # empieza lstm
                        model = dae.lstm
                        for sd_lstm in [0.2, 0.3, 0.4]:
                            for learning_rate_lstm in [5e-6, 5e-4]:
                                for hidden_size2 in [50, 125, 800]:
                                    for hidden_size3 in [10, 25, 70]:
                                        for batch_size_lstm in [5, 10, 20]:
                                            for n_lstm in [0, 10, 50]:

                                                sobre_entrenamiento_lstm = False
                                                for it in range(10):

                                                    if req_grad_dae:
                                                        pass
                                                    else:
                                                        for name, param in model.named_parameters():
                                                            if param.requires_grad:
                                                                param.requires_grad = False
                                                    lstmsuper = LSTMsuper(input_size, hidden_size_dae, hidden_size2,
                                                                          hidden_size3, num_layers, output_size_lstm,
                                                                          batch_size_lstm,
                                                                          model).to(device)
                                                    listLoss_train_valence = []
                                                    listLoss_train_arousal = []
                                                    listLoss_valid_valence = []
                                                    listLoss_valid_arousal = []
                                                    listLoss_valid = []
                                                    listLoss_train = []
                                                    best_loss_valid_mean = np.inf
                                                    i_best_loss_valid_mean = []
                                                    listBest_loss_valid_mean = []

                                                    # (C y D) Entrenamiento y validación modelo lstm
                                                    start_time_lstm = time.time()
                                                    criterion = nn.MSELoss(reduction='mean')
                                                    optimizer = torch.optim.Adam(lstmsuper.parameters(),
                                                                                 lr=learning_rate_lstm)

                                                    for epoch in range(num_epoch):
                                                        listLoss_valid_valence_epoch = []
                                                        listLoss_valid_arousal_epoch = []
                                                        listLoss_train_valence_epoch = []
                                                        listLoss_train_arousal_epoch = []
                                                        listLoss_valid_epoch = []
                                                        lstmsuper.h, lstmsuper.c = lstmsuper.init_hidden(
                                                            lstmsuper.hidden_size)
                                                        lstmsuper.h2, lstmsuper.c2 = lstmsuper.init_hidden(
                                                            lstmsuper.hidden_size2)
                                                        lstmsuper.h3, lstmsuper.c3 = lstmsuper.init_hidden(
                                                            lstmsuper.hidden_size3)
                                                        # batch=0

                                                        # Loss[epoch]={}

                                                        # Lossval[epoch]={}
                                                        for val_fold in np.linspace(1, len(
                                                                features_crossval_corto_standarized), len(
                                                            features_crossval_corto_standarized)):

                                                            val_fold = int(val_fold)

                                                            # Loss[epoch][val_fold]=[]

                                                            list_train_feature_sep, list_valid_feature, list_train_valence_sep, list_valid_valence, list_train_arousal_sep, list_valid_arousal = datatrain_super(
                                                                features_crossval_corto_standarized,
                                                                valence_crossval_corto_standarized,
                                                                arousal_crossval_corto_standarized, val_fold,
                                                                separar_train_folds=True)
                                                            for foldindex in range(len(list_train_feature_sep)):
                                                                # print(foldindex)
                                                                dataset_train_lstm = featData_super(
                                                                    list_train_feature_sep[foldindex],
                                                                    list_train_valence_sep[foldindex],
                                                                    list_train_arousal_sep[foldindex])
                                                                trainloader_lstm = DataLoader(
                                                                    dataset=dataset_train_lstm,
                                                                    batch_size=batch_size_lstm,
                                                                    collate_fn=my_collate_super,
                                                                    drop_last=True, shuffle=True)
                                                                for features_list, valence_list, arousal_list in trainloader_lstm:
                                                                    features_list = normalizar_data(features_list)
                                                                    valence_list = normalizar_data(valence_list)
                                                                    arousal_list = normalizar_data(arousal_list)
                                                                    features_list = corromper_batch(features_list,
                                                                                                    sd_lstm, n_lstm)
                                                                    valence_pack = packInput(valence_list).to(device)
                                                                    arousal_pack = packInput(arousal_list).to(device)
                                                                    annotations_target = torch.cat(
                                                                        [valence_pack.data, arousal_pack.data])
                                                                    valence_out, arousal_out = lstmsuper(
                                                                        packInput(features_list).to(device))
                                                                    annotations_out = torch.cat(
                                                                        [valence_out.data, arousal_out.data])
                                                                    loss_train = criterion(annotations_out,
                                                                                           annotations_target)
                                                                    loss_train_valence = criterion(valence_out.data,
                                                                                                   valence_pack.data)
                                                                    loss_train_arousal = criterion(arousal_out.data,
                                                                                                   arousal_pack.data)
                                                                    optimizer.zero_grad()
                                                                    loss_train.backward()
                                                                    lstmsuper.h = lstmsuper.h.detach()
                                                                    lstmsuper.c = lstmsuper.c.detach()
                                                                    lstmsuper.h2 = lstmsuper.h2.detach()
                                                                    lstmsuper.c2 = lstmsuper.c2.detach()
                                                                    lstmsuper.h3 = lstmsuper.h3.detach()
                                                                    lstmsuper.c3 = lstmsuper.c3.detach()
                                                                    listLoss_train.append(np.sqrt(loss_train.item()))
                                                                    # listLoss_train_epoch.append(np.sqrt(loss_train.item()))
                                                                    listLoss_train_valence.append(
                                                                        np.sqrt(loss_train_valence.item()))
                                                                    listLoss_train_arousal.append(
                                                                        np.sqrt(loss_train_arousal.item()))
                                                                    listLoss_train_valence_epoch.append(
                                                                        np.sqrt(loss_train_valence.item()))
                                                                    listLoss_train_arousal_epoch.append(
                                                                        np.sqrt(loss_train_arousal.item()))

                                                                    optimizer.step()
                                                            with torch.no_grad():
                                                                dataset_valid_lstm = featData_super(list_valid_feature,
                                                                                                    list_valid_valence,
                                                                                                    list_valid_arousal)
                                                                trainloader_valid_lstm = DataLoader(
                                                                    dataset=dataset_valid_lstm,
                                                                    batch_size=batch_size_lstm,
                                                                    collate_fn=my_collate_super,
                                                                    drop_last=True, shuffle=True)

                                                                for features_list, valence_list, arousal_list in trainloader_valid_lstm:
                                                                    features_list = normalizar_data(features_list)
                                                                    valence_list = normalizar_data(valence_list)
                                                                    arousal_list = normalizar_data(arousal_list)
                                                                    valence_pack = packInput(valence_list).to(device)
                                                                    arousal_pack = packInput(arousal_list).to(device)
                                                                    annotations_target = torch.cat(
                                                                        [valence_pack.data, arousal_pack.data])
                                                                    valence_out, arousal_out = lstmsuper(
                                                                        packInput(features_list).to(device))
                                                                    annotations_out = torch.cat(
                                                                        [valence_out.data, arousal_out.data])
                                                                    loss_valid = criterion(annotations_out,
                                                                                           annotations_target)
                                                                    loss_valid_valence = criterion(valence_out.data,
                                                                                                   valence_pack.data)
                                                                    loss_valid_arousal = criterion(arousal_out.data,
                                                                                                   arousal_pack.data)
                                                                    listLoss_valid.append(np.sqrt(loss_valid.item()))
                                                                    listLoss_valid_valence.append(
                                                                        np.sqrt(loss_valid_valence.item()))
                                                                    listLoss_valid_arousal.append(
                                                                        np.sqrt(loss_valid_arousal.item()))
                                                                    listLoss_valid_valence_epoch.append(
                                                                        np.sqrt(loss_valid_valence.item()))
                                                                    listLoss_valid_arousal_epoch.append(
                                                                        np.sqrt(loss_valid_arousal.item()))
                                                                    listLoss_valid_epoch.append(
                                                                        np.sqrt(loss_valid.item()))

                                                            # if val_fold in [1,4,8]:
                                                            # elapsed_time=datetime.timedelta(seconds=time.time()-start_time)
                                                            # print('época: ',epoch,'val_fold=',val_fold,'Loss_train= ',loss_train.item(),'Loss_val= ',loss_valid.item(),'Tiempo transcurrido: ',elapsed_time)
                                                        elapsed_time_lstm = datetime.timedelta(
                                                            seconds=time.time() - start_time_lstm)
                                                        loss_valid_valence_mean = np.mean(listLoss_valid_valence_epoch)
                                                        loss_valid_arousal_mean = np.mean(listLoss_valid_arousal_epoch)
                                                        loss_train_valence_mean = np.mean(listLoss_train_valence_epoch)
                                                        loss_train_arousal_mean = np.mean(listLoss_train_arousal_epoch)
                                                        loss_valid_mean = np.mean(listLoss_valid_epoch)
                                                        # loss_train_mean=np.mean(listLoss_train_epoch)
                                                        if epoch % 5 == 0:
                                                            if loss_valid_mean < best_loss_valid_mean:
                                                                best_loss_valid_mean = loss_valid_mean
                                                                listBest_loss_valid_mean.append(loss_valid_mean)
                                                                i_best_loss_valid_mean.append(epoch)
                                                        print('época: ', epoch, 'Loss_train_valence_mean= ',
                                                              loss_train_valence_mean,
                                                              'Loss_train_arousal_mean= ', loss_train_arousal_mean,
                                                              'Loss_valid_valence_mean=',
                                                              loss_valid_valence_mean, 'Loss_valid_arousal_mean',
                                                              loss_valid_arousal_mean,
                                                              'Tiemo transcurrido: ', elapsed_time_lstm)
                                                    elapsed_time_train_lstm = datetime.timedelta(
                                                        seconds=time.time() - start_time_lstm)
                                                    # Testeo del modelo
                                                    listLoss_test = []
                                                    listLoss_test_valence = []
                                                    listLoss_test_arousal = []
                                                    lstmsuper.change_batch_size(batch_size_test)
                                                    lstmsuper.h, lstmsuper.c = lstmsuper.init_hidden(
                                                        lstmsuper.hidden_size)
                                                    lstmsuper.h2, lstmsuper.c2 = lstmsuper.init_hidden(
                                                        lstmsuper.hidden_size2)
                                                    lstmsuper.h3, lstmsuper.c3 = lstmsuper.init_hidden(
                                                        lstmsuper.hidden_size3)

                                                    # lstmsuper.dae.change_batch_size(batch_size_test)
                                                    list_test_feature, list_test_valence, list_test_arousal = datatrain_test_super(
                                                        features_test_corto_standarized, valence_test_corto_standarized,
                                                        arousal_test_corto_standarized)
                                                    with torch.no_grad():
                                                        dataset_test_lstm = featData_super(list_test_feature,
                                                                                           list_test_valence,
                                                                                           list_test_arousal)
                                                        testloader_test_lstm = DataLoader(dataset=dataset_test_lstm,
                                                                                          batch_size=batch_size_test,
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
                                                            valence_out, arousal_out = lstmsuper(
                                                                packInput(features_list).to(device))
                                                            annotations_out = torch.cat(
                                                                [valence_out.data, arousal_out.data])
                                                            loss_test = criterion(annotations_out, annotations_target)
                                                            loss_test_valence = criterion(valence_out.data,
                                                                                          valence_pack.data)
                                                            loss_test_arousal = criterion(arousal_out.data,
                                                                                          arousal_pack.data)
                                                            listLoss_test.append(loss_test.item())
                                                            listLoss_test_valence.append(
                                                                np.sqrt(loss_test_valence.item()))
                                                            listLoss_test_arousal.append(
                                                                np.sqrt(loss_test_arousal.item()))

                                                    loss_test_valence_mean = np.mean(listLoss_test_valence)
                                                    loss_test_arousal_mean = np.mean(listLoss_test_arousal)
                                                    loss_test_valence_std = np.std(listLoss_test_valence)
                                                    loss_test_arousal_std = np.std(listLoss_test_arousal)
                                                    elapsed_time = datetime.timedelta(seconds=time.time() - start_time)
                                                    model = lstmsuper.dae
                                                    schema_infolstm = {
                                                        'Loss_test_mean_valence': loss_test_valence_mean,
                                                        'Loss_test_std_valence': loss_test_valence_std,
                                                        'Loss_test_mean_arousal': loss_test_arousal_mean,
                                                        'Loss_test_std_arousal': loss_test_arousal_std,
                                                        'Loss_valid_mean_valence': loss_valid_valence_mean,
                                                        'Loss_train_mean_valence': loss_train_valence_mean,
                                                        'Loss_valid_mean_arousal': loss_valid_arousal_mean,
                                                        'Loss_train_mean_arousal': loss_train_arousal_mean,
                                                        'epochs': num_epoch,
                                                        'sd': sd_lstm,
                                                        'batch_size': batch_size_lstm,
                                                        'batch_size_test': batch_size_test,
                                                        'hidden_size': hidden_size_dae,
                                                        'hidden_size2': hidden_size2,
                                                        'hidden_size3': hidden_size3,
                                                        'LR': learning_rate_lstm,
                                                        'tiempo_train': str(elapsed_time_train),
                                                        'tiempo_total': str(elapsed_time),
                                                        'sobre_entrenamiento': sobre_entrenamiento_lstm,
                                                        'superestructura': 1,  # DONDE SACO ESTE DATO?,
                                                        'evol_loss_valid_valence': pickle.dumps(listLoss_valid_valence),
                                                        'evol_loss_valid_arousal': pickle.dumps(listLoss_valid_arousal),
                                                        'evol_loss_valid': pickle.dumps(listLoss_valid),
                                                        'evol_loss_train_valence': pickle.dumps(listLoss_train_valence),
                                                        'evol_loss_train_arousal': pickle.dumps(listLoss_train_arousal),
                                                        'evol_loss_train': pickle.dumps(listLoss_train),
                                                        'list_best_loss_valid_mean': pickle.dumps(
                                                            listBest_loss_valid_mean),
                                                        'index_best_loss_valid_mean': pickle.dumps(
                                                            i_best_loss_valid_mean),
                                                        'ModeloLSTMsuper': pickle.dumps(lstmsuper.state_dict()),
                                                        'comentario': commentary,
                                                        'req_grad_dae': req_grad_dae,
                                                        'pbadae': pba,
                                                        'n': n_lstm,
                                                        'it': it
                                                    }
                                                    sobre_entrenamiento_lstm = f'Loss_test_mean_valenc= {loss_test_valence_mean}, Loss_test_std_valence= {loss_test_valence_std}'
                                                    data = DatosSQL(p_sql)
                                                    data.guardar(schema_infolstm)
                                                    print('corrio')
