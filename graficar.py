from guardar import DatosSQL
from entity import p_sql
import seaborn as sns
from matplotlib import pyplot as plt

data = DatosSQL(p_sql)
# print(device)
panda_sql = data.tabla_df('info_completa_corta')
# plt.figure()
# plt.plot(panda_sql_lstm['LR'])
# plt.show()
# plt.close('all')
# print('chao')
sns.set(style="ticks", color_codes=True)
# pd_data = panda_sql_lstm[
#   ['Loss_test_mean_valence', 'Loss_test_mean_arousal', 'sd', 'epochs', 'n', 'LR', 'hidden_size2', 'hidden_size3',
#    'hidden_size', 'batch_size']]
# plt.figure()
# g = sns.pairplot(pd_data, y_vars=['Loss_test_mean_valence', 'Loss_test_mean_arousal'],
#                 x_vars=['sd', 'epochs', 'n', 'hidden_size2', 'hidden_size3', 'hidden_size', 'LR','batch_size'])
g = sns.pairplot(panda_sql,
                 y_vars=['Loss_test_mean_lstm', 'Loss_test_valence_mean_lstm', 'Loss_test_arousal_mean_lstm'],
                 x_vars=['req_grad_dae', 'n_lstm', 'batch_size_lstm', 'hidden_size3_lstm', 'hidden_size2_lstm',
                         'LR_lstm', 'sd_lstm'])
plt.show()
'''
# plt.show() 

panda_sql_dae = data.tabla_df('infodae')
# panda_sql_lstm.columns
#plt.figure()
#na que ver
pd_data = panda_sql_dae[['Loss_test_mean', 'Loss_test_mean', 'epochs', 'sd', 'batch_size', 'hidden_size', 'LR', 'n']]
g = sns.pairplot(pd_data, y_vars=['Loss_test_mean', 'Loss_test_mean'],
                 x_vars=['sd', 'epochs', 'n', 'LR', 'hidden_size'])
plt.show()
'''
