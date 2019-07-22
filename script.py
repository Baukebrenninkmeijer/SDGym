# from synthetic_data_benchmark.synthesizer.tgan_synthesizer import *
# from synthetic_data_benchmark.synthesizer.tablegan_synthesizer import *
from comet_ml import Experiment
from sdgym.synthesizers.tgan import *
from sdgym.synthesizers.tablegan import *
from sdgym.synthesizers.medgan import *
import pandas as pd
import numpy as np
import json
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

output = 'model_checkpoints'
working_dir = "{}/ckpt_{}".format(output, dataset)
epochs = 100
n = 100000
project_name = "dsgym-tgan"
store_epoch = [1] + list(range(10, epochs + 10, 10))


data = pd.read_csv(f'../data/{dataset}/{dataset}_sdgym.csv')
meta = json.load(open(f'data/real/{dataset}.json', 'r'))

print(f'\nDataset: {dataset} \nEpochs: {epochs}')

synthesizers = dict()
if 'tablegan' in arg_synths:
    synthesizers['tablegan'] = TableganSynthesizer(store_epoch=store_epoch)
if 'tgan' in arg_synths:
    synthesizers['tgan'] = TGANSynthesizer(store_epoch=store_epoch)
if 'medgan' in arg_synths:
    synthesizers['medgan'] = MedganSynthesizer(store_epoch=store_epoch, pretrain_epoch=50)

for synth_name, synthesizer in synthesizers.items():
    synthesizer.init(meta, working_dir)

    experiment = Experiment(api_key=config['comet_ml']['api_key'],
                            project_name=project_name, workspace="baukebrenninkmeijer")
    experiment.log_parameter('dataset', dataset)

    print(f'Training {synth_name}')
    synthesizer.train(data.values, experiment=experiment)

    generated = synthesizer.generate(n)

    data_path = f'generated_data/{dataset}/{synth_name}'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        
    # Loop through all samples
    for i, sample in enumerate(generated):
        z = pd.DataFrame(sample[1])
        epoch = sample[0]
        z.columns = data.columns
        experiment.log_html(z.head(25).to_html())
        if i != epochs:
            z = z[:50]

        if os.path.exists('/mnt'):
            if not os.path.exists('/mnt/samples'):
                os.mkdir('/mnt/samples')
            # z.to_csv(f'/mnt/samples/{dataset}_sample_{project_name}.csv', index=False)
            z.to_csv(f'/mnt/samples/sample_{dataset}_{synth_name}_{epoch}.csv', index=False)
        else:
            if not os.path.exists('samples'):
                os.mkdir('samples')
            z.to_csv(f'samples/sample_{dataset}_{synth_name}_{epoch}.csv', index=False)


    experiment.end()
    print('Done.')
