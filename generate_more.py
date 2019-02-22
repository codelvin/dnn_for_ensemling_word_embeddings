import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

df_g = pd.read_csv('glove_more.csv', sep=' ', header=None, index_col=0, quoting=csv.QUOTE_NONE)
df_g = df_g.iloc[range(0, len(df_g), 2)]
df_s = pd.read_csv('skip_more.csv', sep=' ', header=None, index_col=0, quoting=csv.QUOTE_NONE)
df_s = df_s.iloc[range(0, len(df_s), 2)]

df_g_idx = set(df_g.index)
df_s_idx = set(df_s.index)
inter = df_g_idx.intersection(df_s_idx)
df = pd.DataFrame()

for each in tqdm(inter):
    df = df.append(pd.Series(np.hstack((each, df_g.loc[each][1], df_g.loc[each][2:302], df_s.loc[each][2:302], \
                                                                df_g.loc[each][302:602], df_s.loc[each][302:602], df_g.loc[each][-9:]))), ignore_index=True)
# for i in tqdm(range(len(df_g_idx))):
#     for j in range(len(df_s_idx)):
#         if df_g_idx[i] == df_s_idx[j] and np.argmax(df_g_val[i][-9:]) == np.argmax(df_s_val[j][-9:]):
#             print('haha')
#             df = df.append(pd.Series(np.hstack((df_g_idx[i], df_g_val[i][:300], df_s_val[j][:300], df_g_val[i][300:600], df_s_val[j][300:600], df_g_val[i][-9:]))), ignore_index=True)

print (df.shape)

df.to_csv('more.csv', sep=' ', index = False, header=False, encoding='utf-8')