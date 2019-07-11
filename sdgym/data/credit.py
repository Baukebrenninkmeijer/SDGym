# Generate credit datasets

import json
import os

import numpy as np
import pandas as pd
import sys
sys.path.append('')
from sdgym.utils import CATEGORICAL, CONTINUOUS, verify

output_dir = "data/real/"
temp_dir = "tmp/"


if __name__ == "__main__":
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)

#     df = pd.read_csv("data/raw/creditcard.csv")
    df = pd.read_csv("../data/creditcard/creditcard_num.csv")
    df.drop(columns=['Time'], inplace=True)
    values = df.values

    meta = []
    for i in range(28):
        meta.append({
            "name": "V%d" % i,
            "type": CONTINUOUS,
            "min": np.min(values[:, i]),
            "max": np.max(values[:, i])
        })
    meta.append({
        "name": "Amount",
        "type": CONTINUOUS,
        "min": np.min(values[:, 28]),
        "max": np.max(values[:, 28])
    })
    meta.append({
        "name": "label",
        "type": CATEGORICAL,
        "size": 2,
        "i2s": ["0", "1"]
    })

    np.random.seed(0)
    np.random.shuffle(values)
    t_train = values[:-20000].astype('float32')
    t_test = values[-20000:].astype('float32')

    name = "creditcard"

    pd.DataFrame(values).to_csv(f'../data/{name}/{name}_sdgym.csv', index=False)
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)

    verify("{}/{}.npz".format(output_dir, name), "{}/{}.json".format(output_dir, name))
