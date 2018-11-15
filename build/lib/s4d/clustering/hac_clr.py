import numpy as np
import logging
import copy
from sidekit import Mixture, FeaturesServer
from s4d.clustering.hac_utils import argmin, roll
from s4d.diar import Diar
from sidekit.statserver import StatServer
from bottleneck import argpartition

class HAC_CLR:
    """
    CLR Hierarchical Agglomerative Clustering (HAC) with GMM trained by MAP
    """
    def __init__(self, features_server, diar, ubm, ce=False, ntop=5):
        assert isinstance(features_server, FeaturesServer), 'First parameter has to be a FeatureServer'
        assert isinstance(diar, Diar), '2sd parameter has to be a Diar (segmentationContener)'
        assert isinstance(ubm, Mixture), '3rd parameter has to be a Mixture'

        self.features_server = features_server
        self.diar = copy.deepcopy(diar)
        self.merge = []
        self.nb_merge = 0
        self.ubm = ubm
        self.ce = ce
        self.stat_speaker = None
        self.stat_seg = None
        self.llr = None
        self.ntop = ntop
        #self.init_train()
        #self._init_distance()

    def _get_cep(self, map, cluster):
        cep_list = list()
        for show in map[cluster]:
            idx = self.diar.features_by_cluster(show)[cluster]
            if len(idx) > 0:
                tmp, vad = self.features_server.load(show)
                cep_list.append(tmp[0][idx])
        cep = np.concatenate(cep_list, axis=0)
        return cep

    def _ll(self, ubm, cep, mu=None, name='ubm', argtop = None):
        # ajouter le top gaussien
        lp = ubm.compute_log_posterior_probabilities(cep, mu=mu)

        if argtop is None:
            #logging.info('compute argtop '+speaker)
            argtop = argpartition(lp*-1.0 , self.ntop, axis=1)[:, :self.ntop]
            #logging.info(argtop.shape)
        if self.ntop is not None:
            #logging.info('use ntop '+speaker)
            #logging.info(argtop.shape)
            #logging.info(lp.shape)
            lp = lp[np.arange(argtop.shape[0])[:, np.newaxis], argtop]

        # ppMax = numpy.max(lp, axis=1)

        ll = np.log(np.sum(np.exp(lp), axis=1))
        # ll = ppMax + numpy.log(numpy.sum(numpy.exp((lp.transpose() - ppMax).transpose()),
        #                    axis=1))
        not_finite = np.logical_not(np.isfinite(ll))
        cpt = np.count_nonzero(not_finite)
        # ll[finite] = numpy.finfo('d').min
        ll[not_finite] = 1.0e-200
        m = np.mean(ll)
        if cpt > 0:
            logging.info('model ' + name + '), nb trame with llk problem: %d/%d \t %f', cpt, cep.shape[0], m)
        return m, argtop

    def initial_models(self, nb_threads=1):
        # sort by show to minimize the reading of mfcc by the statServer
        self.diar.sort(['show'])
        # Compute statistics by segments
        self.stat_seg = StatServer(self.diar.id_map())
        self.stat_seg.accumulate_stat(self.ubm, self.features_server)
        self.stat_speaker = self.stat_seg.adapt_mean_MAP_multisession(self.ubm)

    def initial_distances(self, nb_threads=1):
        map = self.diar.make_index(['cluster', 'show'])
        nb = self.stat_speaker.modelset.shape[0]

        self.llr = np.full((nb, nb), np.nan)
        self.dist = np.full((nb, nb), np.nan)
        for i, name_i in enumerate(self.stat_speaker.modelset):
            cep_i = self._get_cep(map, name_i)
            argtop = None
            ll_ubm = None
            if self.ntop is not None or self.ce == False:
                ll_ubm, argtop = self._ll(self.ubm, cep_i, argtop=argtop)

            # self.merge.append([])
            for j, name_j in enumerate(self.stat_speaker.modelset):
                mu = self.stat_speaker.get_model_stat1_by_index(j)
                # if i == 0:
                #    logging.debug(mu)
                self.llr[i, j], _ = self._ll(self.ubm, cep_i, mu=mu, name=name_j, argtop=argtop)
            if self.ce:
                self.llr[i,:] -= self.llr[i,i]
            else:
                self.llr[i,:] -= ll_ubm

        # logging.debug(self.llr)
        self.dist = (self.llr + self.llr.T)*-1.0
        np.fill_diagonal(self.dist, np.finfo('d').max)

    def update(self, i, j, nb_threads=1):
        name_i = self.stat_speaker.modelset[i]
        name_j = self.stat_speaker.modelset[j]
        # logging.debug('%d %d / %s %s', i, j, name_i, name_j)

        for k in range(len(self.stat_seg.modelset)):
            if self.stat_seg.modelset[k] == name_j:
                self.stat_seg.modelset[k] = name_i

        self.stat_speaker = self.stat_seg.adapt_mean_MAP_multisession(self.ubm)

        self.llr = roll(self.llr, j)

        self.diar.rename('cluster', [name_j], name_i)
        map = self.diar.make_index(['cluster', 'show'])
        cep_i = self._get_cep(map, name_i)
        argtop = None
        ll_ubm = None
        if self.ntop > 0 or self.ce == False:
            ll_ubm, argtop = self._ll(self.ubm, cep_i, argtop=argtop)
        for k, name_k in enumerate(self.stat_speaker.modelset):
            mu = self.stat_speaker.get_model_stat1_by_index(k)
            self.llr[i, k], _ = self._ll(self.ubm, cep_i, mu=mu, name=name_k)
        if self.ce:
            self.llr[i,:] -= self.llr[i,i]
        else:
            self.llr[i,:] -= ll_ubm

        self.dist = (self.llr + self.llr.T)*-1.0
        np.fill_diagonal(self.dist, np.finfo('d').max)

    def information(self, i, j, value):
        models = self.stat_speaker.modelset
        self.merge.append([self.nb_merge, models[i], models[j], value])

    def perform(self, thr = 0.0, to_the_end=False):
        models = self.stat_speaker.modelset
        nb = len(models)
        self.nb_merge = -1
        for i in range(nb):
            self.information(i, i, 0)

        i, j, v = argmin(self.dist, nb)
        self.nb_merge = 0
        while v < thr and nb > 1:
            self.information(i ,j, v)
            self.nb_merge += 1
            logging.debug('merge: %d c1: %s (%d) c2: %s (%d) dist: %f',
                          self.nb_merge, models[i], i, models[j], j, v)
            # update merge
            # update model and distance
            self.update(i, j)
            nb -= 1
            i, j, v = argmin(self.dist, nb)

        end_diar = copy.deepcopy(self.diar)
        if to_the_end:
            while nb > 1:
                self.information(i ,j, v)
                self.nb_merge += 1
                logging.debug('merge: %d c1: %s (%d) c2: %s (%d) dist: %f',
                              self.nb_merge, models[i], i, models[j], j, v)
                # update merge
                # update model and distance
                self.update(i, j)
                nb -= 1
                i, j, v = argmin(self.dist, nb)

        return end_diar



