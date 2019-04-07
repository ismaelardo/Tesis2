# importar todo
from entityCorto import *
from guardar import DatosSQL

req_grad_dae = 0
output_size_lstm = 1
num_epoch_dae = 10
num_epoch_lstm = 10
input_size = 260
output_size_dae = 260
num_layers = 1
batch_size_test_dae = 10
batch_size_test_lstm = 10
pba = 0
criterion = nn.MSELoss(reduction='mean')

# iterador
for sd_dae in [0.2, 0.3, 0.4]:
    for learning_rate_dae in [5e-6, 5e-4]:
        for hidden_size_dae in [50, 200, 800]:
            for batch_size_dae in [5, 10, 20]:
                for n_dae in [0, 10, 50]:
                    parametros_dae = {'input_size_dae': input_size, 'output_size_dae': output_size_dae,
                                      'sd_dae': sd_dae,
                                      'LR_dae': learning_rate_dae, 'hidden_size_dae': hidden_size_dae,
                                      'batch_size_dae': batch_size_dae, 'batch_size_test_dae': batch_size_test_dae,
                                      'n_dae': n_dae, 'criterion_dae': criterion}
                    dae = DAE(parametros_dae).to(device)
                    epochs_dae = 0
                    for it_dae in range(10):
                        for epoch_dae in range(num_epoch_dae):
                            epochs_dae += 1
                            dae.h, dae.c = dae.init_hidden()
                            if epoch_dae == num_epoch_dae - 1:
                                Loss_valid_dae = []
                            for val_fold_dae in np.linspace(1, len(features_crossval_standarized),
                                                            len(features_crossval_standarized)):
                                val_fold_dae = int(val_fold_dae)
                                list_train_sep_dae, list_valid_dae = datatrain_dae(features_crossval_standarized,
                                                                                   val_fold_dae,
                                                                                   separar_train_folds=True)
                                # entrenamiento
                                dae.train(list_train_sep_dae)
                                # validacion
                                if epoch_dae == num_epoch_dae - 1:
                                    Loss_valid_dae.append(dae.valid(list_valid_dae))
                            if epoch_dae == num_epoch_dae - 1:
                                Loss_valid_mean_dae = np.mean(Loss_valid_dae)
                        dae.batch_size = batch_size_test_dae
                        dae.h, dae.c = dae.init_hidden()
                        list_test_feature_dae = datatrain_test_dae(features_test_standarized)
                        Loss_test_mean_dae = dae.test(list_test_feature_dae)


                        model = dae.lstm
                        for sd_lstm in [0.2, 0.3, 0.4]:
                            for learning_rate_lstm in [5e-6, 5e-4]:
                                for hidden_size2 in [50, 125, 800]:
                                    for hidden_size3 in [10, 25, 70]:
                                        for batch_size_lstm in [5, 10, 20]:
                                            for n_lstm in [0, 10, 50]:
                                                for req_grad_dae in [0, 1]:
                                                    parametros_lstm = {'input_size_lstm': input_size,
                                                                       'output_size_lstm': output_size_lstm,
                                                                       'hidden_size_lstm': hidden_size_dae,
                                                                       'hidden_size2_lstm': hidden_size2,
                                                                       'hidden_size3_lstm': hidden_size3,
                                                                       'n_lstm': n_lstm,
                                                                       'sd_lstm': sd_lstm,
                                                                       'LR_lstm': learning_rate_lstm,
                                                                       'batch_size_lstm': batch_size_lstm,
                                                                       'batch_size_test_lstm': batch_size_test_lstm,
                                                                       'criterion_lstm': criterion}
                                                    epochs_lstm = 0
                                                    for it_lstm in range(10):

                                                        if req_grad_dae:
                                                            pass
                                                        else:
                                                            for name, param in model.named_parameters():
                                                                if param.requires_grad:
                                                                    param.requires_grad = False
                                                        lstmsuper = LSTMsuper(parametros_lstm,
                                                                              model).to(device)

                                                        for epoch_lstm in range(num_epoch_lstm):
                                                            epochs_lstm = epochs_lstm + 1
                                                            lstmsuper.h, lstmsuper.c = lstmsuper.init_hidden(
                                                                lstmsuper.hidden_size)
                                                            lstmsuper.h2, lstmsuper.c2 = lstmsuper.init_hidden(
                                                                lstmsuper.hidden_size2)
                                                            lstmsuper.h3, lstmsuper.c3 = lstmsuper.init_hidden(
                                                                lstmsuper.hidden_size3)
                                                            if epoch_lstm == num_epoch_lstm - 1:
                                                                Loss_valid_lstm = []
                                                                Loss_valid_valence_lstm = []
                                                                Loss_valid_arousal_lstm = []
                                                            for val_fold_lstm in np.linspace(1, len(
                                                                    features_crossval_corto_standarized), len(
                                                                features_crossval_corto_standarized)):
                                                                val_fold_lstm = int(val_fold_lstm)
                                                                list_train_feature_sep_lstm, list_train_valence_sep_lstm, list_train_arousal_sep_lstm, list_valid_feature_lstm, list_valid_valence_lstm, list_valid_arousal_lstm = datatrain_super(
                                                                    features_crossval_corto_standarized,
                                                                    valence_crossval_corto_standarized,
                                                                    arousal_crossval_corto_standarized, val_fold_lstm,
                                                                    separar_train_folds=True)
                                                                # entrenamiento
                                                                lstmsuper.train(list_train_feature_sep_lstm,
                                                                                list_train_valence_sep_lstm,
                                                                                list_train_arousal_sep_lstm)
                                                                # validacion
                                                                if epoch_lstm == num_epoch_lstm - 1:
                                                                    Loss_valid_val, Loss_valid_valence_val, Loss_valid_arousal_val = lstmsuper.valid(
                                                                        list_valid_feature_lstm,
                                                                        list_valid_valence_lstm,
                                                                        list_valid_arousal_lstm)
                                                                    Loss_valid_lstm.append(Loss_valid_val)
                                                                    Loss_valid_valence_lstm.append(
                                                                        Loss_valid_valence_val)
                                                                    Loss_valid_arousal_lstm.append(
                                                                        Loss_valid_arousal_val)
                                                            if epoch_lstm == num_epoch_lstm - 1:
                                                                print(f'len(Loss_valid_lstm): {len(Loss_valid_lstm)}')
                                                                Loss_valid_lstm=list(np.concatenate(Loss_valid_lstm))
                                                                Loss_valid_valence_lstm=list(np.concatenate(Loss_valid_valence_lstm))
                                                                Loss_valid_arousal_lstm=list(np.concatenate(Loss_valid_arousal_lstm))
                                                                Loss_valid_mean_lstm = np.mean(Loss_valid_lstm)
                                                                Loss_valid_valence_mean_lstm = np.mean(
                                                                    Loss_valid_valence_lstm)
                                                                Loss_valid_arousal_mean_lstm = np.mean(
                                                                    Loss_valid_arousal_lstm)
                                                        # Testeo del modelo
                                                        listLoss_test, listLoss_test_valence, listLoss_test_arousal = lstmsuper.test(
                                                            features_test_corto_standarized,
                                                            valence_test_corto_standarized,
                                                            arousal_test_corto_standarized)

                                                        # lstmsuper.dae.change_batch_size(batch_size_test)
                                                        Loss_test_mean_lstm = np.mean(listLoss_test)
                                                        Loss_test_valence_mean_lstm = np.mean(listLoss_test_valence)
                                                        Loss_test_arousal_mean_lstm = np.mean(listLoss_test_arousal)
                                                        parametros_dae['criterion_dae']=str(parametros_dae['criterion_dae'])
                                                        parametros_lstm['criterion_lstm']=str(parametros_lstm['criterion_lstm'])

                                                        schema_info_corto = {**parametros_dae, **parametros_lstm, 'tabla_corta': 1,
                                                             'Loss_test_mean_lstm': float(Loss_test_mean_lstm),
                                                             'Loss_test_valence_mean_lstm': float(Loss_test_valence_mean_lstm),
                                                             'Loss_test_arousal_mean_lstm': float(Loss_test_arousal_mean_lstm),
                                                             'Loss_valid_mean_lstm': float(Loss_valid_mean_lstm),
                                                             'Loss_valid_valence_mean_lstm': float(Loss_valid_valence_mean_lstm),
                                                             'Loss_valid_arousal_mean_lstm': float(Loss_valid_arousal_mean_lstm),
                                                             'req_grad_dae': req_grad_dae,
                                                             'Loss_valid_mean_dae': float(Loss_valid_mean_dae),
                                                             'Loss_test_mean_dae': float(Loss_test_mean_dae),
                                                             'epochs_dae': epochs_dae,
                                                             'epochs_lstm': epochs_lstm
                                                        }

                                                        data = DatosSQL(p_sql)
                                                        data.guardar(schema_info_corto)
                                                        parametros_dae['criterion_dae'] = criterion
                                                        parametros_lstm['criterion_lstm'] = criterion
                                                        model=lstmsuper.dae
