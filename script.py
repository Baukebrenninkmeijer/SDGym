# from synthetic_data_benchmark.synthesizer.tgan_synthesizer import *
# from synthetic_data_benchmark.synthesizer.tablegan_synthesizer import *
from sdgym.synthesizers.tgan import *
from sdgym.synthesizers.tablegan import *
from sdgym.synthesizers.medgan import *
import pandas as pd
import numpy as np
import json
from comet_ml import Experiment
import configparser
import argparse

parser = argparse.ArgumentParser(description='Evaluate data synthesizers')
parser.add_argument('--dataset', type=str, help='Which dataset to choose. Options are berka, creditcard and ticket')
parser.add_argument('--synthesizers', nargs='*', help='Which synthesizers/generators to use.', default=['tablegan', 'tgan', 'medgan'])

args = parser.parse_args()
dataset = args.dataset
arg_synths = args.synthesizers

config = configparser.ConfigParser()
config.read('config.ini')

output = 'samples/test'
working_dir = "{}/ckpt_{}".format(output, dataset)
epochs = 100
n = 1000000
project_name = "dsgym-tgan"

data = pd.read_csv(f'../data/{dataset}/{dataset}_sdgym.csv')
meta = json.load(open(f'data/real/{dataset}.json', 'r'))

print(f'\nDataset: {dataset} \nEpochs: {epochs}\n')

synthesizers = dict()
if 'tablegan' in arg_synths:
    synthesizers['tablegan'] = TableganSynthesizer(store_epoch=[epochs])
if 'tgan' in arg_synths:
    synthesizers['tgan'] = TGANSynthesizer(store_epoch=[epochs])
if 'medgan' in arg_synths:
    synthesizers['medgan'] = MedganSynthesizer(store_epoch=[epochs], pretrain_epoch=50)

for synth_name, synthesizer in synthesizers.items():
    synthesizer.init(meta, working_dir)

    experiment = Experiment(api_key=config['comet_ml']['api_key'],
                            project_name=project_name, workspace="baukebrenninkmeijer")
    experiment.log_parameter('dataset', dataset)

    print(f'Training {synth_name}')
    synthesizer.train(data.values, experiment=experiment)

    generated = synthesizer.generate(n)

    z = pd.DataFrame(generated[0][1])
    z.columns = data.columns
    data_path = f'generated_data/{dataset}/{synth_name}'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    z.to_csv(f'{data_path}/sample_{epochs}.csv', index=False)

    experiment.log_asset_data(z, file_name=f'sample_{dataset}_{project_name}_{len(z)}', overwrite=False)
    experiment.end()
    print('Done.')
