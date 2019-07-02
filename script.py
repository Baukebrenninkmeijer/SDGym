# from synthetic_data_benchmark.synthesizer.tgan_synthesizer import *
# from synthetic_data_benchmark.synthesizer.tablegan_synthesizer import *
from sdgym.synthesizers.tgan import *
from sdgym.synthesizers.tablegan import *
from sdgym.synthesizers.medgan import *
import pandas as pd
import numpy as np
import json
from comet_ml import Experiment
from tqdm.auto import tqdm

import argparse

parser = argparse.ArgumentParser(description='Evaluate data synthesizers')
parser.add_argument('dataset', type=str, help='Which dataset to choose. Options are berka, creditcard and ticket')

args = parser.parse_args()
dataset = args.dataset

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

output = 'samples/test'
working_dir = "{}/ckpt_{}".format(output, dataset)

# data = pd.read_csv('../data/berka/berka.csv', sep=';')
data = pd.read_csv(f'../data/{dataset}/{dataset}_sdgym.csv')
meta = json.load(open(f'data/real/{dataset}.json', 'r'))

project_name = ""
experiment = Experiment(api_key="49HGMPyIKjokHwg2pVOKWTG67",
                        project_name=project_name, workspace="baukebrenninkmeijer")

epochs = 100
print(f'\nDataset: {dataset} \nEpochs: {epochs}')
synthesizer = TableganSynthesizer(store_epoch=[epochs])
# synthesizer = TableganSynthesizer(store_epoch=[epochs])
synthesizer.init(meta, working_dir)

synthesizer.train(data.values, cometml_key=config['comet_ml']['api_key'])


print('Generating data...')
n = 1000000
generated = synthesizer.generate(n)

z = pd.DataFrame(generated[0][1])
z.columns = data.columns
if not os.path.exists(f'generated_data/{dataset}'):
    os.mkdir(f'generated_data/{dataset}')
z.to_csv(f'generated_data/{dataset}/sample_{epochs}.csv', index=False)

experiment.log_asset_data(z, file_name=f'sample_{ds}_{project_name}_{len(p)}', overwrite=False)
experiment.log_dataset_info(name=ds)
experiment.end()
print('Done.')
