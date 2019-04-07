from entity import *
from guardar import DatosSQL

# lineas que se guardaran como comentario en sql
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
batch_size_test_dae = 10
batch_size_test_lstm = 10
pba_dae = 0
criterion = nn.MSELoss(reduction='mean')
listas_dae_train = {}
listas_dae_valid = {}

for sd_dae in [0.2, 0.3, 0.4]:
    for learning_rate_dae in [5e-6, 5e-4]:
        for hidden_size_dae in [50, 200, 800]:
            for batch_size_dae in [5, 10, 20]:
                for n_dae in [0, 10, 50]:
                    parametros_dae = {'input_size': input_size, 'output_size': output_size_dae, 'sd': sd_dae,
                                      'LR': learning_rate_dae, 'hidden_size': hidden_size_dae,
                                      'batch_size': batch_size_dae, 'batch_size_test': batch_size_test_dae,
                                      'n': n_dae, 'criterion': criterion}
                    dae = DAE(parametros_dae).to(device)
                    # dae.load_state_dict(torch.load(f_estructura+'Dae_loss_test_mean=0.457_loss_test_sd=0.033/'+'ModeloDAE'))
                    sobre_entrenamiento = 0
                    epochs_dae = 0
                    for it_dae in range(10):
                        pba_dae = pba_dae + 1
                        start_time = time.time()
                        listas_dae_train['evol_loss_train'] = []
                        listas_dae_valid['evol_loss_valid'] = []
                        best_loss_valid_mean = np.inf
                        i_best_loss_valid_mean = []
                        list_best_loss_valid_mean = []

                        for epoch in range(num_epoch):
                            epochs_dae += 1
                            listas_dae_train['listLoss_train_epoch'] = []
                            listas_dae_valid['listLoss_valid_epoch'] = []
                            dae.h, dae.c = dae.init_hidden()
                            for val_fold in np.linspace(1, len(features_crossval_standarized),
                                                        len(features_crossval_standarized)):
                                val_fold = int(val_fold)

                                listas_dae_train['list_train_sep'], listas_dae_valid['list_valid'] = datatrain_dae(
                                    features_crossval_standarized, val_fold, separar_train_folds=True)
                                # entrenamiento
                                listas_dae_train = dae.train(listas_dae_train)
                                # validación
                                listas_dae_valid = dae.valid(listas_dae_valid)
                            loss_valid_mean_epoch = float(np.mean(listas_dae_valid['listLoss_valid_epoch']))
                            loss_train_mean_epoch = float(np.mean(listas_dae_train['listLoss_train_epoch']))
                            if epoch % 5 == 0:
                                if loss_valid_mean_epoch < best_loss_valid_mean:
                                    best_loss_valid_mean = loss_valid_mean_epoch
                                    list_best_loss_valid_mean.append(loss_valid_mean_epoch)
                                    i_best_loss_valid_mean.append(epoch)
                        # print(len(listLoss_valid_epoch),len(listLoss_train_epoch))

                        elapsed_time_train = datetime.timedelta(seconds=time.time() - start_time)

                        dae.batch_size = batch_size_test_dae
                        dae.h, dae.c = dae.init_hidden()

                        # lstmsuper.dae.change_batch_size(batch_size_test)
                        list_test_feature = datatrain_test_dae(features_test_standarized)
                        # Testeo del modelo
                        listLoss_test = dae.test(list_test_feature)
                        loss_test_std = float(np.std(listLoss_test))
                        loss_test_mean = np.mean(listLoss_test)

                        dae.batch_size = batch_size_dae
                        elapsed_time = datetime.timedelta(seconds=time.time() - start_time)
                        listas_dae_train['evol_loss_train'] = pickle.dumps(listas_dae_train['evol_loss_train'])
                        listas_dae_valid['evol_loss_valid'] = pickle.dumps(listas_dae_valid['evol_loss_valid'])
                        parametros_dae['criterion'] = str(parametros_dae['criterion'])
                        schema_infodae = {**parametros_dae,
                                          'Loss_test_mean': loss_test_mean,
                                          'Loss_test_std': loss_test_std,
                                          'evol_loss_train': listas_dae_train['evol_loss_train'],
                                          'evol_loss_valid': listas_dae_valid['evol_loss_valid'],
                                          'Loss_valid_mean_last_epoch': loss_valid_mean_epoch,
                                          'Loss_train_mean_last_epoch': loss_train_mean_epoch,
                                          'epochs': num_epoch,
                                          'batch_size_test': batch_size_test_dae,
                                          'tiempo_train': str(elapsed_time_train),
                                          'tiempo_total': str(elapsed_time),
                                          'sobre_entrenamiento': sobre_entrenamiento,
                                          'estructura': 1,  # DONDE SACO ESTE DATO?,
                                          'list_best_loss_valid_mean': pickle.dumps(list_best_loss_valid_mean),
                                          'index_best_loss_valid_mean': pickle.dumps(i_best_loss_valid_mean),
                                          'ModeloDAE': pickle.dumps(dae.state_dict()),
                                          'pba': pba_dae,
                                          'it': it_dae
                                          # 'comentario': commentarydae
                                          }
                        parametros_dae['criterion'] = criterion
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
                        listas_lstm_train = {}
                        listas_lstm_valid = {}
                        pba_lstm = 0
                        for sd_lstm in [0.2, 0.3, 0.4]:
                            for learning_rate_lstm in [5e-6, 5e-4]:
                                for hidden_size2 in [50, 125, 800]:
                                    for hidden_size3 in [10, 25, 70]:
                                        for batch_size_lstm in [5, 10, 20]:
                                            for n_lstm in [0, 10, 50]:
                                                if sd_lstm == 0.2 and learning_rate_lstm == 5e-6 and hidden_size2 in [
                                                    50, 125]:
                                                    pass
                                                else:
                                                    parametros_lstm = {'input_size': input_size,
                                                                       'output_size': output_size_lstm,
                                                                       'hidden_size': hidden_size_dae,
                                                                       'hidden_size2': hidden_size2,
                                                                       'hidden_size3': hidden_size3, 'n': n_lstm,
                                                                       'sd': sd_lstm, 'LR': learning_rate_lstm,
                                                                       'batch_size': batch_size_lstm,
                                                                       'batch_size_test': batch_size_test_lstm,
                                                                       'criterion': criterion}
                                                    sobre_entrenamiento_lstm = False
                                                    epochs_lstm = 0
                                                    for it_lstm in range(10):
                                                        pba_lstm += 1

                                                        if req_grad_dae:
                                                            pass
                                                        else:
                                                            for name, param in model.named_parameters():
                                                                if param.requires_grad:
                                                                    param.requires_grad = False
                                                        lstmsuper = LSTMsuper(parametros_lstm,
                                                                              model).to(device)
                                                        listas_lstm_train['listLoss_train_valence'] = []
                                                        listas_lstm_train['listLoss_train_arousal'] = []
                                                        listas_lstm_valid['listLoss_valid_valence'] = []
                                                        listas_lstm_valid['listLoss_valid_arousal'] = []
                                                        listas_lstm_valid['listLoss_valid'] = []
                                                        listas_lstm_train['listLoss_train'] = []
                                                        best_loss_valid_mean = np.inf
                                                        i_best_loss_valid_mean = []
                                                        listBest_loss_valid_mean = []

                                                        # (C y D) Entrenamiento y validación modelo lstm
                                                        start_time_lstm = time.time()

                                                        for epoch in range(num_epoch):
                                                            epochs_lstm = epochs_lstm + 1
                                                            listas_lstm_valid['listLoss_valid_valence_epoch'] = []
                                                            listas_lstm_valid['listLoss_valid_arousal_epoch'] = []
                                                            listas_lstm_train['listLoss_train_valence_epoch'] = []
                                                            listas_lstm_train['listLoss_train_arousal_epoch'] = []
                                                            listas_lstm_valid['listLoss_valid_epoch'] = []
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

                                                                listas_lstm_train['list_train_feature_sep'], \
                                                                listas_lstm_valid['list_valid_feature'], \
                                                                listas_lstm_train['list_train_valence_sep'], \
                                                                listas_lstm_valid['list_valid_valence'], \
                                                                listas_lstm_train['list_train_arousal_sep'], \
                                                                listas_lstm_valid[
                                                                    'list_valid_arousal'] = datatrain_super(
                                                                    features_crossval_corto_standarized,
                                                                    valence_crossval_corto_standarized,
                                                                    arousal_crossval_corto_standarized, val_fold,
                                                                    separar_train_folds=True)
                                                                # entrenamiento lstm
                                                                listas_lstm_train = lstmsuper.train(listas_lstm_train)
                                                                # validacion lstm
                                                                listas_lstm_valid = lstmsuper.valid(listas_lstm_valid)

                                                                elapsed_time_lstm = datetime.timedelta(
                                                                    seconds=time.time() - start_time_lstm)
                                                            loss_valid_valence_mean = np.mean(
                                                                listas_lstm_valid['listLoss_valid_valence_epoch'])
                                                            loss_valid_arousal_mean = np.mean(
                                                                listas_lstm_valid['listLoss_valid_arousal_epoch'])
                                                            loss_train_valence_mean = np.mean(
                                                                listas_lstm_train['listLoss_train_valence_epoch'])
                                                            loss_train_arousal_mean = np.mean(
                                                                listas_lstm_train['listLoss_train_arousal_epoch'])
                                                            loss_valid_mean = np.mean(
                                                                listas_lstm_valid['listLoss_valid_epoch'])
                                                            # loss_train_mean=np.mean(listLoss_train_epoch)
                                                            if epochs_lstm % 5 == 0:
                                                                if loss_valid_mean < best_loss_valid_mean:
                                                                    best_loss_valid_mean = loss_valid_mean
                                                                    listBest_loss_valid_mean.append(loss_valid_mean)
                                                                    i_best_loss_valid_mean.append(epochs_lstm)
                                                            print('pba_dae', pba_dae, 'pba_lstm', pba_lstm, 'n', n_lstm,
                                                                  'batch_size_lstm',
                                                                  batch_size_lstm,
                                                                  'época: ', epochs_lstm, 'Loss_train_valence_mean= ',
                                                                  loss_train_valence_mean,
                                                                  'Loss_train_arousal_mean= ', loss_train_arousal_mean,
                                                                  'Loss_valid_valence_mean=',
                                                                  loss_valid_valence_mean, 'Loss_valid_arousal_mean',
                                                                  loss_valid_arousal_mean,
                                                                  'Tiemo transcurrido: ', elapsed_time_lstm)
                                                        elapsed_time_train_lstm = datetime.timedelta(
                                                            seconds=time.time() - start_time_lstm)
                                                        # Testeo del modelo
                                                        listLoss_test, listLoss_test_valence, listLoss_test_arousal = lstmsuper.test(
                                                            features_test_corto_standarized,
                                                            valence_test_corto_standarized,
                                                            arousal_test_corto_standarized)

                                                        # lstmsuper.dae.change_batch_size(batch_size_test)

                                                        loss_test_valence_mean = np.mean(listLoss_test_valence)
                                                        loss_test_arousal_mean = np.mean(listLoss_test_arousal)
                                                        loss_test_valence_std = np.std(listLoss_test_valence)
                                                        loss_test_arousal_std = np.std(listLoss_test_arousal)
                                                        elapsed_time = datetime.timedelta(
                                                            seconds=time.time() - start_time)
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
                                                            'epochs': epochs_lstm,
                                                            'sd': sd_lstm,
                                                            'batch_size': batch_size_lstm,
                                                            'batch_size_test': batch_size_test_lstm,
                                                            'hidden_size': hidden_size_dae,
                                                            'hidden_size2': hidden_size2,
                                                            'hidden_size3': hidden_size3,
                                                            'LR': learning_rate_lstm,
                                                            'tiempo_train': str(elapsed_time_train),
                                                            'tiempo_total': str(elapsed_time),
                                                            'sobre_entrenamiento': sobre_entrenamiento_lstm,
                                                            'superestructura': 1,  # DONDE SACO ESTE DATO?,
                                                            'evol_loss_valid_valence': pickle.dumps(
                                                                listas_lstm_valid['listLoss_valid_valence']),
                                                            'evol_loss_valid_arousal': pickle.dumps(
                                                                listas_lstm_valid['listLoss_valid_arousal']),
                                                            'evol_loss_valid': pickle.dumps(
                                                                listas_lstm_valid['listLoss_valid']),
                                                            'evol_loss_train_valence': pickle.dumps(
                                                                listas_lstm_train['listLoss_train_valence']),
                                                            'evol_loss_train_arousal': pickle.dumps(
                                                                listas_lstm_train['listLoss_train_arousal']),
                                                            'evol_loss_train': pickle.dumps(
                                                                listas_lstm_train['listLoss_train']),
                                                            'list_best_loss_valid_mean': pickle.dumps(
                                                                listBest_loss_valid_mean),
                                                            'index_best_loss_valid_mean': pickle.dumps(
                                                                i_best_loss_valid_mean),
                                                            'ModeloLSTMsuper': pickle.dumps(lstmsuper.state_dict()),
                                                            'comentario': commentary,
                                                            'req_grad_dae': req_grad_dae,
                                                            'pba_dae': pba_dae,
                                                            'pba_lstm': pba_lstm,
                                                            'n': n_lstm,
                                                            'it': it_lstm
                                                        }
                                                        sobre_entrenamiento_lstm = f'Loss_test_mean_valenc= {loss_test_valence_mean}, Loss_test_std_valence= {loss_test_valence_std}'
                                                        data = DatosSQL(p_sql)
                                                        data.guardar(schema_infolstm)
                                                        print('corrio')
