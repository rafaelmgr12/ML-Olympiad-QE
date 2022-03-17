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

notas = ['NU_NOTA_CN']

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
    
df['NU_IDADE'] = df['NU_IDADE'].fillna(df['NU_IDADE'].mode().iloc[0])
df['CO_MUNICIPIO_NASCIMENTO'] = df['CO_MUNICIPIO_NASCIMENTO'].fillna(df['CO_MUNICIPIO_PROVA'])
df['CO_MUNICIPIO_ESC'] = df['CO_MUNICIPIO_ESC'].fillna(df['CO_MUNICIPIO_PROVA'])
df['TP_SIT_FUNC_ESC'] = df['TP_SIT_FUNC_ESC'].fillna(1)
df['TP_LOCALIZACAO_ESC'] = df['TP_LOCALIZACAO_ESC'].fillna(1)
df['SG_UF_ESC'] = df['SG_UF_ESC'].fillna(df['SG_UF_NASCIMENTO'])

df_idhm_ifdm = pd.read_csv('https://raw.githubusercontent.com/rrpronaldo/quality_education/main/dataset_idhm_ifdm.csv')
dict_idhm = dict(zip(df_idhm_ifdm.CO_MUNICIPIO,df_idhm_ifdm.VR_IDHM))
df['VR_IDHM'] = df.CO_MUNICIPIO_RESIDENCIA.map(dict_idhm)

df_ifdm = pd.read_csv('https://raw.githubusercontent.com/rrpronaldo/quality_education/main/dataset_idhm_ifdm.csv')
dict_ifdm = dict(zip(df_ifdm.CO_MUNICIPIO,df_ifdm.IFDM_2010))
df['VR_IFDM'] = df.CO_MUNICIPIO_RESIDENCIA.map(dict_ifdm)
    
## FEATURE ENGINEERING


## --- variáveis para codificar
list_toenc = np.array(['Q001','Q002', 'Q003', 'Q004', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011','Q012', 'Q013',
            'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021','Q022', 'Q023', 'Q024', 'Q025',
            'CO_MUNICIPIO_RESIDENCIA', 'CO_ESCOLA', 'CO_MUNICIPIO_PROVA',
            'SG_UF_PROVA', 'SG_UF_NASCIMENTO', 'SG_UF_ESC','TP_LINGUA','TP_SEXO',
            'CO_MUNICIPIO_ESC', 'CO_MUNICIPIO_NASCIMENTO', 'CO_UF_NASCIMENTO', 'CO_UF_PROVA'])

## --- variáveis mais relevantes para matemática
#ft_cn = np.array(['TP_PRESENCA_CN', 'TP_STATUS_REDACAO', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'Q006', 'NU_IDADE', 'TP_ST_CONCLUSAO', 'Q024', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'Q008', 'Q002', 'Q010', 'Q003', 'TP_DEPENDENCIA_ADM_ESC', 'CO_ESCOLA', 'IN_TREINEIRO', 'TP_LINGUA', 'Q018', 'Q004', 'Q001', 'Q025', 'Q022', 'Q007', 'CO_MUNICIPIO_PROVA', 'CO_UF_PROVA', 'TP_SIT_FUNC_ESC', 'Q013', 'CO_MUNICIPIO_RESIDENCIA', 'CO_MUNICIPIO_ESC', 'Q016', 'SG_UF_ESC', 'Q009', 'TP_ENSINO', 'Q014', 'CO_MUNICIPIO_NASCIMENTO', 'CO_UF_NASCIMENTO', 'TP_LOCALIZACAO_ESC', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'Q019', 'TP_COR_RACA', 'Q017', 'Q021', 'Q023', 'Q005', 'SG_UF_PROVA', 'SG_UF_NASCIMENTO', 'Q020', 'Q012', 'Q011', 'IN_TEMPO_ADICIONAL', 'TP_NACIONALIDADE','Q015'])

ft_cn = list_vars

ft_clust = ['TP_PRESENCA_CN', 'Q006', 'NU_IDADE', 'Q024', 'TP_ESCOLA', 'Q004', 'Q002', 'Q003']


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
        np.save('./Encoders/natureza/classes_%s.npy' % n, enc.classes_)
else:
    encoders = []
    for n in np.arange(len(list_toenc)):
        enc = LabelEncoder()
        enc.classes_ = np.load('./Encoders/natureza/classes_%s.npy' % n)
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
#df['NU_IDADE'] = df['NU_IDADE'].apply(lambda x: x if x<25 else 25)

df['Q006'] = df['Q006'].apply(lambda x: x if x<11 else 11)


#df['Q004'] = df['Q004'].apply(lambda x: 0 if x==0 else
#                                        1 if (x==1) | (x==2) | (x==5) else x-1)

df['Q002'] = df['Q002'].apply(lambda x: 1 if (x==1) | (x==7) else x)

#df['TP_ANO_CONCLUIU'] = df['TP_ANO_CONCLUIU'].apply(lambda x: x if x<3 else 3)

df['Q003'] = df['Q003'].apply(lambda x: 0 if x==0 else 1 if (x==1) | (x==2) | (x==5) else x-1)

## TREINO DO MODELO

## --- instanciamento do agrupador
nclust = 3
model = make_pipeline(StandardScaler(), KMeans(n_clusters=nclust))
model.fit(df[ft_clust])

# atribui cada amostra a um grupo
df['CLUSTER'] = model.predict(df[ft_clust])

regressor_cn = make_pipeline(StandardScaler(), lgbm.LGBMRegressor(boosting_type='gbdt', 
                                                                   learning_rate=0.1, 
                                                                   max_depth=-1, 
                                                                   n_estimators=350))
## --- fit do modelo e gravação dos parâmetros em arquivo
regressor_cn.fit(df[np.append(ft_cn, ['CLUSTER','VR_IDHM','VR_IFDM'])], 
                 df['NU_NOTA_CN'])
pickle.dump(model, open('./models/natureza.sav', 'wb'))

del df

## PREDICAO DAS NOTAS DE CIE. DA NATUREZA

submissions = pd.read_csv('../input/test.csv', engine='c', usecols=['NU_INSCRICAO'])
df_cn = reduce_mem_usage( pd.read_csv('../input/test.csv', engine='c', usecols=ft_cn) )

df_cn['NU_IDADE'] = df_cn['NU_IDADE'].fillna(df_cn['NU_IDADE'].mode().iloc[0])
df_cn['CO_MUNICIPIO_NASCIMENTO'] = df_cn['CO_MUNICIPIO_NASCIMENTO'].fillna(df_cn['CO_MUNICIPIO_PROVA'])
df_cn['CO_MUNICIPIO_ESC'] = df_cn['CO_MUNICIPIO_ESC'].fillna(df_cn['CO_MUNICIPIO_PROVA'])
df_cn['TP_SIT_FUNC_ESC'] = df_cn['TP_SIT_FUNC_ESC'].fillna(1)
df_cn['TP_LOCALIZACAO_ESC'] = df_cn['TP_LOCALIZACAO_ESC'].fillna(1)
df_cn['SG_UF_ESC'] = df_cn['SG_UF_ESC'].fillna(df_cn['SG_UF_NASCIMENTO'])

df_cn['VR_IDHM'] = df_cn.CO_MUNICIPIO_RESIDENCIA.map(dict_idhm)
df_cn['VR_IFDM'] = df_cn.CO_MUNICIPIO_RESIDENCIA.map(dict_ifdm)
    

## --- substitui valores NaN (variáveis numéricas)  por inteiro arbitrário 

for coluna in df_cn.loc[:2,~df_cn.columns.isin(list_toenc)].columns:
    df_cn[coluna] = df_cn[coluna].fillna(-32768).astype('int16')
    
i=0
for coluna in list_toenc:
    df_cn[coluna] = df_cn[coluna].astype(str).fillna("missing").replace("nan", "missing").astype('category')
    df_cn[coluna] = encoders[i].transform(df_cn[coluna])
    i+=1
    
## --- redução de cardinalidade
#df_cn['NU_IDADE'] = df_cn['NU_IDADE'].apply(lambda x: x if x<25 else 25)

df_cn['Q006'] = df_cn['Q006'].apply(lambda x: x if x<11 else 11)


#df_cn['Q004'] = df_cn['Q004'].apply(lambda x: 0 if x==0 else
#                                        1 if (x==1) | (x==2) | (x==5) else x-1)

df_cn['Q002'] = df_cn['Q002'].apply(lambda x: 1 if (x==1) | (x==7) else x)

#df_cn['TP_ANO_CONCLUIU'] = df_cn['TP_ANO_CONCLUIU'].apply(lambda x: x if x<3 else 3)

df_cn['Q003'] = df_cn['Q003'].apply(lambda x: 0 if x==0 else
                                        1 if (x==1) | (x==2) | (x==5) else x-1)

## clusterização
df_cn['CLUSTER'] = model.predict(df_cn[ft_clust])
## predição
submissions['NU_NOTA_CN'] = regressor_cn.predict(df_cn[np.append(ft_cn, ['CLUSTER','VR_IDHM','VR_IFDM'])])
submissions['NU_NOTA_CN'].iloc[df_cn[df_cn['TP_PRESENCA_CN']!=1].index] = 0

submissions.to_csv('./submissions.csv', index=False)
