# from synthetic_data_benchmark.synthesizer.tgan_synthesizer import *
from synthetic_data_benchmark.synthesizer.tablegan_synthesizer import *
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

output = 'samples/test'
dataset = 'berka'
working_dir = "{}/ckpt_{}".format(output, dataset)

data = pd.read_csv('../data/berka/berka.csv', sep=';')[:1000000]

import pickle
subs = pickle.load(open('../data/berka/subs.pkl', 'rb'))

meta = [
    {
    'name': 'account_id',
    'type': 'continuous',
    'min': data.account_id.min(),
    'max': data.account_id.max()
    },
#     {
#     'name': 'account_id',
#     'type': 'categorical',
#     'size': len(data.account_id.unique().tolist()),
#     'i2s': subs['account_id']['label'].values.tolist()
#     },
    {
    'name': 'trans_amount',
    'type': 'continuous',
    'min': data.trans_amount.min(),
    'max': data.trans_amount.max()   
    },
    {
    'name': 'balance_after_trans',
    'type': 'continuous',
    'min': data.balance_after_trans.min(),
    'max': data.balance_after_trans.max()    
    },
    {
    'name': 'trans_type',
    'type': 'categorical',
    'size': 3,
    'i2s': ['CREDIT', 'WITHDRAWAL', 'UNKNOWN']    
    },
    {
    'name': 'trans_operation',
    'type': 'categorical',
    'size': 6,
    'i2s': subs['trans_operation']['label'].values.tolist()    
    },
#     {
#     'name': 'trans_date',
#     'type': 'ordinal',
#     'size': len(data.trans_date.unique().tolist()),
#     'i2s': list(range(data.trans_date.max()))    
#     },
    {
    'name': 'trans_date',
    'type': 'continuous',
    'size': data.trans_date.min(),
    'i2s': data.trans_date.max()    
    },
]

# synthesizer = TGANSynthesizer(store_epoch=[100])
synthesizer = TableganSynthesizer(store_epoch=[100])
synthesizer.init(meta, working_dir)

synthesizer.train(data.values, cometml_key=config['comet_ml']['api_key'])


print('Generating data...')
n = 1000000
generated = synthesizer.generate(n)

z = pd.DataFrame(generated[0][1])
z.columns = data.columns
z.to_csv(f'generated_data/{dataset}/sample_{epoch}.csv')
print('Done.')
