## --- Bibliotecas para estrutura de dados
import pandas as pd
import numpy as np

## --- Funções de pre-processamento
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

## --- Bibliotecas de machine learning
from sklearn.cluster import KMeans
import lightgbm as lgbm

## --- Funções definidas pelo usuário
from subroutines import reduce_mem_usage

import warnings
import pickle
warnings.filterwarnings('ignore')

## variáveis relevantes para leitura

## --- notas

notas = ['NU_NOTA_MT']

### --- variáveis gerais
list_vars = np.array(['Q001','Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011','Q012', 'Q013',
            'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021','Q022', 'Q023', 'Q024', 'Q025',
            'IN_ACESSO', 'TP_ANO_CONCLUIU','TP_SEXO', 'TP_DEPENDENCIA_ADM_ESC','TP_LINGUA',
            'NU_IDADE', 'TP_ESCOLA', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'IN_LIBRAS',
            'CO_MUNICIPIO_RESIDENCIA', 'CO_ESCOLA', 'CO_MUNICIPIO_PROVA',
            'TP_ENSINO', 'SG_UF_PROVA', 'TP_ESTADO_CIVIL', 'TP_NACIONALIDADE',
            'IN_SEM_RECURSO', 'IN_SALA_ESPECIAL', 'SG_UF_NASCIMENTO', 'SG_UF_ESC',
            'IN_TREINEIRO', 'IN_DEFICIT_ATENCAO', 'TP_SIT_FUNC_ESC',
            'CO_MUNICIPIO_ESC', 'IN_LEDOR', 'IN_TEMPO_ADICIONAL',
            'IN_DEFICIENCIA_AUDITIVA', 'TP_LOCALIZACAO_ESC', 'IN_DEFICIENCIA_MENTAL',
            'IN_SURDEZ', 'IN_AUTISMO', 'IN_DEFICIENCIA_FISICA', 'IN_TRANSCRICAO',
            'CO_MUNICIPIO_NASCIMENTO', 'CO_UF_NASCIMENTO', 'CO_UF_PROVA',
            'IN_MAQUINA_BRAILE', 'TP_PRESENCA_MT', 'TP_PRESENCA_LC',
            'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_STATUS_REDACAO'])

## --- lendo dados de treino
reader = pd.read_csv('../input/train.csv', engine='c', chunksize=50000, 
                     nrows=2500000, usecols=np.append(list_vars, notas) )

df = pd.DataFrame(columns=pd.read_csv('../input/train.csv', nrows=2, 
                                      usecols=np.append(list_vars, notas)).columns)
for chunk in reader:
    df = pd.concat([df ,reduce_mem_usage(chunk)])
    
df_idhm_ifdm = pd.read_csv('https://raw.githubusercontent.com/rrpronaldo/quality_education/main/dataset_idhm_ifdm.csv')
dict_idhm = dict(zip(df_idhm_ifdm.CO_MUNICIPIO,df_idhm_ifdm.VR_IDHM))
df['VR_IDHM'] = df.CO_MUNICIPIO_RESIDENCIA.map(dict_idhm)

df_ifdm = pd.read_csv('https://raw.githubusercontent.com/rrpronaldo/quality_education/main/dataset_idhm_ifdm.csv')
dict_ifdm = dict(zip(df_ifdm.CO_MUNICIPIO,df_ifdm.IFDM_2010))
df['VR_IFDM'] = df.CO_MUNICIPIO_RESIDENCIA.map(dict_ifdm)
    
## --- variáveis para codificar
list_toenc = np.array(['Q006', 'Q024', 'Q008', 'Q003', 'Q004', 'TP_LINGUA', 'Q010',
                       'Q002', 'Q018', 'Q007', 'Q013', 'TP_SEXO', 'Q001', 'Q019', 'Q016',
                       'Q021', 'Q014', 'Q022', 'CO_ESCOLA', 'Q025','Q005',
                       'CO_MUNICIPIO_RESIDENCIA', 'CO_UF_PROVA', 'CO_MUNICIPIO_PROVA',
                       'Q009', 'CO_MUNICIPIO_ESC', 'CO_UF_NASCIMENTO', 'SG_UF_ESC',
                       'Q023', 'CO_MUNICIPIO_NASCIMENTO', 'Q017', 'Q012',
                       'SG_UF_NASCIMENTO', 'SG_UF_PROVA', 'Q011', 'Q015', 'Q020'])

## --- variáveis mais relevantes para matemática
#ft_mt = np.array(['TP_PRESENCA_MT', 'TP_PRESENCA_CN', 'Q006', 'Q024', 'NU_IDADE', 'Q008', 'Q003', 'Q004', 'TP_LINGUA', 'TP_ST_CONCLUSAO', 'Q010', 'Q002', 'TP_ESCOLA', 'TP_ANO_CONCLUIU', 'Q018', 'TP_DEPENDENCIA_ADM_ESC','Q007', 'Q013','TP_SEXO', 'Q001', 'Q019', 'Q016', 'Q021', 'Q014', 'Q022', 'CO_ESCOLA', 'TP_COR_RACA', 'Q025', 'CO_MUNICIPIO_RESIDENCIA', 'CO_UF_PROVA', 'CO_MUNICIPIO_PROVA', 'Q009', 'IN_TREINEIRO', 'CO_MUNICIPIO_ESC', 'TP_ENSINO', 'CO_UF_NASCIMENTO', 'SG_UF_ESC', 'Q023', 'CO_MUNICIPIO_NASCIMENTO', 'TP_SIT_FUNC_ESC', 'Q017', 'TP_ESTADO_CIVIL', 'Q012', 'SG_UF_NASCIMENTO', 'TP_LOCALIZACAO_ESC', 'Q005', 'SG_UF_PROVA', 'Q011', 'TP_NACIONALIDADE', 'Q015', 'Q020','IN_ACESSO', 'IN_LIBRAS', 'IN_SEM_RECURSO', 'IN_SALA_ESPECIAL'] )

ft_mt = list_vars

ft_clust = ['Q006', 'Q024', 'Q008', 'Q003']

## --- encoder
is_to_enc = 0
if is_to_enc == 1:
    ##encoding
    enc1 = reduce_mem_usage( pd.read_csv('../input/train.csv', engine='c',
                                       usecols=list_toenc) )
    enc2 = reduce_mem_usage( pd.read_csv('../input/test.csv', engine='c',
                                       usecols=list_toenc) )

    enc = pd.concat([enc1,enc2])
    del enc1, enc2
    encoders = []
    for coluna in list_toenc:
        if enc[coluna].isna().any() == True:
            encoders.append( LabelEncoder().fit( list(set(enc[coluna].astype(str).fillna("missing").replace("nan", "missing").unique().tolist())) ) )
        else:
            encoders.append( LabelEncoder().fit( list(set(enc[coluna].astype(str).unique().tolist())) ) )
    del enc
    ## saving encoders
    for enc, n in zip( encoders, np.arange(len(encoders)) ):
        np.save('./Encoders/matematica/classes_%s.npy' % n, enc.classes_)
else:
    encoders = []
    for n in np.arange(len(list_toenc)):
        enc = LabelEncoder()
        enc.classes_ = np.load('./Encoders/matematica/classes_%s.npy' % n)
        encoders.append(enc)
        
## --- substitui notas missing por zero
for coluna in notas:
    df[coluna] = df[coluna].fillna(0)
    
## --- substitui valores NaN (variáveis numéricas)  por inteiro arbitrário 

for coluna in df[list_vars].loc[:2,~df[list_vars].columns.isin(list_toenc)].columns:
    df[coluna] = df[coluna].fillna(-32768).astype('int16')
    
i=0
for coluna in list_toenc:
    df[coluna] = df[coluna].astype(str).fillna("missing").replace("nan", "missing").astype('category')
    df[coluna] = encoders[i].transform(df[coluna])
    i+=1
    
## --- redução de cardinalidade
#df['NU_IDADE'] = df['NU_IDADE'].apply(lambda x: x if x<35 else 35)

#df['Q004'] = df['Q004'].apply(lambda x: 0 if x==0 else
#                                        1 if (x==1) | (x==2) | (x==5) else x-1)

df['Q002'] = df['Q002'].apply(lambda x: 1 if (x==1) | (x==7) else x)

#df['TP_ANO_CONCLUIU'] = df['TP_ANO_CONCLUIU'].apply(lambda x: x if x<4 else 4)

df['Q003'] = df['Q003'].apply(lambda x: 0 if x==0 else
                                        1 if (x==1) | (x==2) | (x==5) else x-1)

## TREINO DO MODELO

## --- instanciamento do agrupador
nclust = 5
model = make_pipeline(StandardScaler(), KMeans(n_clusters=nclust))
model.fit(df[ft_clust])

# atribui cada amostra a um grupo
df['CLUSTER'] = model.predict(df[ft_clust])

regressor_mt = make_pipeline(StandardScaler(), lgbm.LGBMRegressor(boosting_type='gbdt', 
                                                                   learning_rate=0.15, 
                                                                   max_depth=-1, 
                                                                   n_estimators=250))
## --- fit do modelo e gravação dos parâmetros em arquivo
regressor_mt.fit(df[np.append(ft_mt,['CLUSTER','VR_IFDM', 'VR_IDHM'])], 
                 df['NU_NOTA_MT'])
pickle.dump(model, open('./models/matematica.sav', 'wb'))

del df

## PREDIÇÂO DAS NOTAS DE MATEMÁTICA

submissions = pd.read_csv('./submissions.csv', engine='c')
df_mt = reduce_mem_usage( pd.read_csv('../input/test.csv', engine='c', usecols=list_vars) )

df_mt['VR_IDHM'] = df_mt.CO_MUNICIPIO_RESIDENCIA.map(dict_idhm)
df_mt['VR_IFDM'] = df_mt.CO_MUNICIPIO_RESIDENCIA.map(dict_ifdm)

## --- substitui valores NaN (variáveis numéricas)  por inteiro arbitrário 

for coluna in df_mt.loc[:2,~df_mt.columns.isin(list_toenc)].columns:
    df_mt[coluna] = df_mt[coluna].fillna(-32768).astype('int16')
    
i=0
for coluna in list_toenc:
    df_mt[coluna] = df_mt[coluna].astype(str).fillna("missing").replace("nan", "missing").astype('category')
    df_mt[coluna] = encoders[i].transform(df_mt[coluna])
    i+=1
    
## --- redução de cardinalidade
#df_mt['NU_IDADE'] = df_mt['NU_IDADE'].apply(lambda x: x if x<35 else 35)

#df_mt['Q004'] = df_mt['Q004'].apply(lambda x: 0 if x==0 else
#                                        1 if (x==1) | (x==2) | (x==5) else x-1)

df_mt['Q002'] = df_mt['Q002'].apply(lambda x: 1 if (x==1) | (x==7) else x)

#df_mt['TP_ANO_CONCLUIU'] = df_mt['TP_ANO_CONCLUIU'].apply(lambda x: x if x<4 else 4)

df_mt['Q003'] = df_mt['Q003'].apply(lambda x: 0 if x==0 else
                                        1 if (x==1) | (x==2) | (x==5) else x-1)

## clusterização
df_mt['CLUSTER'] = model.predict(df_mt[ft_clust])
## predição
submissions['NU_NOTA_MT'] = regressor_mt.predict(df_mt[np.append(ft_mt,['CLUSTER','VR_IFDM', 'VR_IDHM'])])
submissions['NU_NOTA_MT'].iloc[df_mt[df_mt['TP_PRESENCA_MT']!=1].index] = 0
submissions.to_csv('./submissions.csv', index=False)

