__author__ = 'meignier'

import numpy as np
import logging
import copy
from math import isnan
from s4d.clustering.hac_utils import argmin, roll
from s4d.clustering.hac_utils import bic_square_root
from s4d.clustering.gauss import GaussFull
from s4d.diar import Diar

class HAC_BIC:
    """
    BIC Hierarchical Agglomerative Clustering (HAC) with gaussian models

    The algorithm is based upon a hierarchical agglomerative clustering. The
    initial set of clusters is composed of one segment per cluster. Each cluster
    is modeled by a Gaussian with a full covariance matrix (see
    :class:`gauss.GaussFull`). :math:`\Delta BIC`
    measure is employed to select the candidate clusters to group as well as
    to stop the merging process. The two closest clusters :math:`i` and
    :math:`j` are merged at each iteration until :math:`\\Delta BIC_{i,j} > 0`.


        :math:`\\Delta BIC_{i,j} = PBIC_{i+j} - PBIC_{i} - PBIC_{j} - P`

        :math:`PBIC_{x}  = \\frac{n_x}{2} \\log|\\Sigma_x|`

        :math:`cst  = \\frac{1}{2} \\alpha \\left(d + \\frac{d(d+1)}{2}\\right)`

        :math:`P  = cst + log(n_i+n_j)`

    where :math:`|\\Sigma_i|`, :math:`|\\Sigma_j|` and :math:`|\\Sigma|` are the
    determinants of gaussians associated to the clusters :math:`i`, :math:`j`
    and :math:`i+j`. :math:`\\alpha` is a parameter to set up. The penalty
    factor :math:`P` depends on :math:`d`, the dimension of the features, as well as
    on :math:`n_i` and :math:`n_j`, refering to the total length of cluster
    :math:`i` and cluster :math:`j` respectively.


    """
    def __init__(self, cep, table, alpha=1.0, sr=False):
        self.cep = cep
        self.dim = cep.shape[1];
        self.alpha = alpha
        self.diar = copy.deepcopy(table)
        self.models = []
        self.merge = []
        self.nb_merge = 0
        self.sr = sr
        self.dist = None
        self._init_train()
        self._init_distance()

    def _init_train(self):
        """
        Train initial models

        """
        map = self.diar.make_index(['cluster'])
        for cluster in map:
            model = GaussFull(cluster, self.dim)
            self.models.append(model)
            self.cst_bic = GaussFull.cst_bic(self.dim, self.alpha)
            for row in map[cluster]:
                start = row['start']
                stop = row['stop']
                model.add(self.cep[start:stop])

        for model in self.models:
            model.compute()

    def _init_distance(self):
        """ Compute distance matrix
        """
        nb = len(self.models)
        self.dist = np.full((nb, nb), np.nan)
        # for i in range(0, nb):
        #    mi = self.models[i]
        for i, mi in enumerate(self.models):
            # self.merge.append([])
            # for j, mj in enumerate(self.models, start=i+1):
            #    logging.debug('i %d j %d n %d', i, j ,nb)
            for j in range(i + 1, nb):
                mj = self.models[j]
                self.dist[i, j] = self.dist[j, i] = self._dist(mi, mj)
        #logging.debug(self.dist)

    def _dist(self, mi, mj):
        """
        Compute the BIC distance d(i,j)
        :param mi: a GaussFull object
        :param mj: a GaussFull object
        :return: float
        """
        v = GaussFull.merge_partial_bic(mi, mj) - mi.partial_bic - mj.partial_bic
        if self.sr:
            v += - bic_square_root(mi.count, mj.count, self.alpha, self.dim)
        else :
            v += - self.cst_bic * np.log(mi.count + mj.count)
        if isnan(v):
            logging.warning('BIC is NAN, mi: '+mi.name+' ' + str(mi.count)+' mj: '+mj.name+' ' + str(mj.count))
        return v

    def _merge_model(self, mi, mj):
        """
        Merge two a GaussFull objects
        :param mi: a GaussFull object
        :param mj: a GaussFull object
        :return: a GaussFull object
        """
        return GaussFull.merge(mi, mj)

    def _update_dist(self, i):
        """
        Update row and column i of the distance matrix
        :param i: int

        """
        nb = len(self.models)
        mi = self.models[i]
        for j in (x for x in range(nb) if x != i):
            mj = self.models[j]
            self.dist[i, j] = self.dist[j, i] = self._dist(mi, mj)

    def information(self, i, j, value, duration):
        self.merge.append([self.nb_merge, self.models[i].name, self.models[j].name, value, duration])

    def perform(self, to_the_end=False):
        """
        perform the HAC algorithm
        :return: a Diar object and a dictonary mapping the old cluster_list to the
        new lables
        """
        nb = len(self.models)
        self.nb_merge = -1
        for i in range(nb):
            self.information(i, i, 0, self.models[i].count)

        i, j, v = argmin(self.dist, nb)
        self.nb_merge = 0
        while v < 0.0 and nb > 1:
            self.information(i, j, v, self.models[i].count+self.models[j].count)
            self.nb_merge += 1
            logging.debug('merge: %d c1: %s (%d) c2: %s (%d) dist: %f %d',
                          self.nb_merge, self.models[i].name, i,
                          self.models[j].name, j, v, nb)
            # update merge
            # self.merge[i].append(
            #    [self.nb_merge, self.models[i].speaker, self.models[j].speaker, v])
            # self.merge[i] += self.merge[j]
            # self.merge.pop(j)
            self.diar.rename('cluster', [self.models[j].name], self.models[i].name)
            # update model
            self.models[i] = self._merge_model(self.models[i], self.models[j])
            self.models.pop(j)
            # nb = len(self.models)
            # update distances
            self.dist = roll(self.dist, j)
            self._update_dist(i)
            nb -= 1
            i, j, v = argmin(self.dist, nb)

        out_diar = copy.deepcopy(self.diar)
        n = self.nb_merge

        if to_the_end:
            while nb > 1:
                self.information(i, j, v, self.models[i].count+self.models[j].count)
                self.nb_merge += 1
                logging.debug('merge: %d c1: %s (%d) c2: %s (%d) dist: %f %d',
                          self.nb_merge, self.models[i].name, i,
                          self.models[j].name, j, v, nb)
                self.diar.rename('cluster', [self.models[j].name], self.models[i].name)
                # update model
                self.models[i] = self._merge_model(self.models[i], self.models[j])
                self.models.pop(j)
                # nb = len(self.models)
                # update distances
                self.dist = roll(self.dist, j)
                self._update_dist(i)
                nb -= 1
                i, j, v = argmin(self.dist, nb)

        return out_diar


def hac_bic(feature_server, diar, threshold, square_root_bic = False):
    shows = diar.make_index(['show'])
    diar_out = Diar()
    for show in shows:
        cep, _ = feature_server.load(show)
        bic = HAC_BIC(cep, shows[show], alpha=threshold, sr=square_root_bic)
        diar_out += bic.perform(to_the_end=True)
    return diar_out
