from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np

from ..utils import CATEGORICAL, ORDINAL, CONTINUOUS

class DiscretizeTransformer(object):
    """Discretize continuous columns into several bins.
    Transformation result is a int array."""
    def __init__(self, meta, n_bins):
        self.meta = meta
        self.c_index = [id for id, info in enumerate(meta) if info['type'] == CONTINUOUS]
        self.kbin_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    def fit(self, data):
        if self.c_index == []:
            return
        self.kbin_discretizer.fit(data[:, self.c_index])

    def transform(self, data):
        if self.c_index == []:
            return data.astype('int')

        data_t = data.copy()
        data_t[:, self.c_index] = self.kbin_discretizer.transform(data[:, self.c_index])
        return data_t.astype('int')

    def inverse_transform(self, data):
        if self.c_index == []:
            return data

        data_t = data.copy().astype('float32')
        data_t[:, self.c_index] = self.kbin_discretizer.inverse_transform(data[:, self.c_index])
        return data_t

class GeneralTransformer(object):
    """
    Continuous and ordinal columns are normalized to [0, 1].
    Discrete columns are converted to a one-hot vector.
    """
    def __init__(self, meta, act='sigmoid'):
        self.act = act
        self.meta = meta
        self.output_dim = 0
        for info in self.meta:
            if info['type'] in [CONTINUOUS, ORDINAL]:
                self.output_dim += 1
            else:
                self.output_dim += info['size']

    def fit(self, data):
        pass

    def transform(self, data):
        data_t = []
        self.output_info = []
        for id_, info in enumerate(self.meta):
            col = data[:, id_]
            if info['type'] == CONTINUOUS:
                col = (col - (info['min'])) / (info['max'] - info['min'])
                if self.act == 'tanh':
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))
            elif info['type'] == ORDINAL:
                col = col / info['size']
                if self.act == 'tanh':
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))
            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), col.astype('int32')] = 1
                data_t.append(col_t)
                self.output_info.append((info['size'], 'softmax'))
        return np.concatenate(data_t, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])

        data = data.copy()
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                current = data[:, 0]
                data = data[:, 1:]


                if self.act == 'tanh':
                    current = (current + 1) / 2
                current = np.clip(current, 0, 1)
                data_t[:, id_] = current * (info['max'] - info['min']) + info['min']

            elif info['type'] == ORDINAL:
                current = data[:, 0]
                data = data[:, 1:]
                if self.act == 'tanh':
                    current = (current + 1) / 2
                current = current * info['size']
                current = np.round(current).clip(0, info['size'] - 1)
                data_t[:, id_] = current
            else:
                current = data[:, :info['size']]
                data = data[:, info['size']:]
                data_t[:, id_] = np.argmax(current, axis=1)

        return data_t

class GMMTransformer(object):
    """
    Continuous columns are modeled with a GMM.
        and then normalized to a scalor [0, 1] and a n_cluster dimensional vector.

    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, meta, n_clusters=5):
        self.meta = meta
        self.n_clusters = n_clusters

    def fit(self, data):
        model = []

        self.output_info = []
        self.output_dim = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                gm = GaussianMixture(self.n_clusters)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                self.output_info += [(1, 'tanh'), (self.n_clusters, 'softmax')]
                self.output_dim += 1 + self.n_clusters
            else:
                model.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == CONTINUOUS:
                current = current.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (2 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                argmax = np.argmax(probs, axis=1)
                idx = np.arange((len(features)))
                features = features[idx, argmax].reshape([-1, 1])

                features = np.clip(features, -.99, .99)

                values += [features, probs]
            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), current.astype('int32')] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st+1:st+1+self.n_clusters]
                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)
                u = np.clip(u, -1, 1)
                st += 1 + self.n_clusters
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 2 * std_t  + mean_t
                data_t[:, id_] = tmp
            else:
                current = data[:, st:st+info['size']]
                st += info['size']
                data_t[:, id_] = np.argmax(current, axis=1)
        return data_t

class BGMTransformer(object):
    """
    Continuous columns are modeled with a BayesianGMM.
        and then normalized to a scalor [0, 1] and a vector.

    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, meta, n_clusters=10, eps=0.005):
        """n_cluster is the upper bound of modes
        """
        self.meta = meta
        self.n_clusters = n_clusters
        self.eps = eps

    def fit(self, data):
        model = []

        self.output_info = []
        self.output_dim = 0
        self.components = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                gm = BayesianGaussianMixture(self.n_clusters,
                        weight_concentration_prior_type='dirichlet_process',
                        weight_concentration_prior = 0.001,
                        n_init=1)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                comp = gm.weights_ > self.eps
                self.components.append(comp)
                print(np.sum(comp))
                self.output_info += [(1, 'tanh'), (np.sum(comp), 'softmax')]
                self.output_dim += 1 + np.sum(comp)
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == CONTINUOUS:
                current = current.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (4 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))

                n_opts = sum(self.components[id_])
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(data), dtype='int')
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)

                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                values += [features, probs_onehot]
            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), current.astype('int32')] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st+1:st+1+np.sum(self.components[id_])]
                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)
                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t  + mean_t
                data_t[:, id_] = tmp
            else:
                current = data[:, st:st+info['size']]
                st += info['size']
                data_t[:, id_] = np.argmax(current, axis=1)
        return data_t
