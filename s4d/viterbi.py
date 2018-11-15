# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#
# This file is part of S4D.
#
# SD4 is a python package for speaker diarization based on SIDEKIT.
# S4D home page: http://www-lium.univ-lemans.fr/s4d/
# SIDEKIT home page: http://www-lium.univ-lemans.fr/sidekit/
#
# S4D is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# S4D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

__author__ = 'meignier'

import logging
from sidekit import Mixture, StatServer, FeaturesServer
from sidekit.mixture import sum_log_probabilities
import numpy
from s4d.diar import Diar
from s4d.utils import FeatureServerFake
import copy


class Viterbi:
    """
    Class that implements a Videtbi decoding and one-state HMM
    """
    eps = 1.0e-200

    def __init__(self, cep, diarization, exit_penalties=[0],
                 loop_penalties=[0]):
        self.cep = cep
        self.diarization = diarization
        shows = self.diarization.unique('show')
        self.cluster_list = self.diarization.unique('cluster')
        self.nb_clusters = len(self.cluster_list)
        if len(shows) > 1:
            raise Exception('diarization indexes serveal shows')

        self.nb_features = self.cep.shape[0]
        self.show = shows.pop()
        self.mixtures = None
        self.names = None
        self.observation = None
        self.transition_probabilities = None
        self.exit_penalties = exit_penalties
        self.loop_penalties = loop_penalties

    # def train_soft(self):
    #     self.mixtures = list()
    #     cluster_eatures = self.diarization.features_by_cluster(index_max=self.nb_features)
    #     # init GMM
    #     for i in range(0, self.nb_clusters):
    #         cluster = self.cluster_list[i];
    #         m = Mixture(name=cluster)
    #         idx = cluster_eatures[cluster]
    #         data = self.cep[idx]
    #         llk = m.EM_uniform(data, 8, 3, 10, llk_gain=0.01)
    #         self.mixtures.append(m)
    #     self._init_transition()
    #
    #     for it in range(1):
    #         # print('emission: ', len(self.mixtures), self.nb_clusters, len(self.cluster_list))
    #         self.emission()
    #         # soft train
    #         # print('soft train: ', len(self.mixtures), self.nb_clusters, len(self.cluster_list))
    #         for i in range(len(self.mixtures)):
    #             # print(it, i, self.cluster_list[i], self.nb_clusters)
    #             m = self.mixtures[i]
    #             # expectation
    #             accum = copy.deepcopy(m)
    #             accum._reset()
    #
    #             for seg in self.diarization:
    #                 #if seg['cluster'] != m.speaker:
    #                 #    continue
    #                 #else:
    #                 #    print(seg['cluster'], m.speaker)
    #                 start = seg['start']
    #                 stop = seg['stop']
    #                 llks = numpy.mean(self.observation[start:stop, :], axis=0)
    #                 w = llks / numpy.sum(llks)
    #                 # w = llks / numpy.max(llks)
    #                 # w = llks / 6.0
    #                 data = self.cep[start:stop]
    #
    #                 lp = m.compute_log_posterior_probabilities(data)
    #                 pp, loglk = sum_log_probabilities(lp)
    #                 pp *= w[i]
    #                 # zero order statistics
    #                 accum.w += pp.sum(0)
    #                 # print('\t', start, stop, accum.w)
    #                 # first order statistics
    #                 accum.mu += numpy.dot(data.T, pp).T
    #                 # second order statistics
    #                 accum.invcov += numpy.dot(numpy.square(data.T), pp).T
    #
    #             m.w = accum.w / numpy.sum(accum.w)
    #             m.mu = accum.mu / accum.w[:, numpy.newaxis]
    #             cov = accum.invcov / accum.w[:, numpy.newaxis] - numpy.square(m.mu)
    #             m.invcov = 1.0 / cov
    #             m._compute_all()

    def train(self, distrib_nb=8, init=None, max_it=4):
        """
        Trains one GMM for each cluster using EM.

        """

        iterations=[1, 2, 4, 4, 8]
        idx = int(numpy.log2(distrib_nb)) - 1
        iterations[idx] = max_it

        self.mixtures = list()
        cluster_features = self.diarization.features_by_cluster(
            maximum_length=self.nb_features)
        new_cluster_list = list()
        self.names = list()
        for i in range(0, self.nb_clusters):
            cluster = self.cluster_list[i]
            index = cluster_features[cluster]
            data = self.cep[index]

            if init is None:
                mixture = Mixture(name=cluster)
                llk = mixture.EM_split(FeatureServerFake(data), [self.show], distrib_nb=distrib_nb, iterations=iterations, llk_gain=0.01, num_thread=1)
            else:
                mixture = init[i]
                if mixture.name != cluster:
                    logging.error("!!! name don't match %s != %s", mixture.name, cluster)
                llk = mixture.EM_no_init(FeatureServerFake(data), [self.show], max_iteration=5,  llk_gain=0.01, num_thread=1)


            #llk = m.EM_uniform(FeatureServerFake(data), [self.show], distrib_nb=distrib_nb, llk_gain=0.01, num_thread=1)
            sum_llk = sum(llk)
            if numpy.isfinite(sum_llk) and sum_llk != 0.0:
                self.mixtures.append(mixture)
                self.names.append(self.mixtures[i].name)
                new_cluster_list.append(cluster)
            else:
                logging.warning('bad model, remove it: ' + cluster + ' '+ str(llk)+ ' nb features: '+str(len(index)))
        self.cluster_list = new_cluster_list
        self.nb_clusters = len(self.cluster_list)
        self._init_transition()

    def _init_transition(self):
        self.transition_probabilities = numpy.full((self.nb_clusters, self.nb_clusters),
                                                   self.exit_penalties[-1], dtype=numpy.int)
        for i in range(0, self.nb_clusters):
            self.transition_probabilities[i, i] = self.loop_penalties[
                min(i, len(self.loop_penalties) - 1)]
            if i < len(self.exit_penalties) - 1:
                for j in range(0, self.nb_clusters):
                    if i != j:
                        self.transition_probabilities[i, j] = self.exit_penalties[
                            min(i, len(self.exit_penalties) - 1)]

    def emission(self):
        """
        Computes the log-likelihood for each features.

        """
        self.observation = numpy.zeros((self.nb_features, self.nb_clusters))
        #corrupt_llk_list = list()
        for i in range(0, self.nb_clusters):
            lp = self.mixtures[i].compute_log_posterior_probabilities(self.cep)
            #self.observation[:, i] = numpy.log(numpy.sum(numpy.exp(lp), axis=1))
            pp_max = numpy.max(lp, axis=1)
            self.observation[:, i]  = pp_max + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).transpose()), axis=1))
            #logging.info("--> %f %f", numpy.mean(self.observation[:, i]), numpy.mean(ll))


        #     finite = numpy.logical_not(numpy.isfinite(self.observation[:, i]))
        #     cpt = numpy.count_nonzero(finite)
        #
        #     if cpt >= self.nb_features/10:
        #         logging.debug('model ' + self.cluster_list[i] + '(' + str(i) + '), nb trame with llk problem: ' + str(cpt) + ' ' + str(self.nb_features))
        #         corrupt_llk_list.append(i)
        #     else:
        #         self.observation[finite, i] = numpy.finfo('d').min
        #
        # if len(corrupt_llk_list) > 0:
        #     for i in reversed(corrupt_llk_list):
        #         del self.cluster_list[i]
        #         del self.mixtures[i]
        #     self.nb_clusters = len(self.cluster_list)
        #
        #     self.observation = numpy.delete(self.observation, corrupt_llk_list, axis=1)
        #     self.transition_probabilities = numpy.delete(self.transition_probabilities, corrupt_llk_list, axis=1)
        #     self.transition_probabilities = numpy.delete(self.transition_probabilities, corrupt_llk_list, axis=0)

    def decode(self, table):
        """
        performs a Viterbi decoding of the segment given in diarization
        :param table: a Diar object
        :return: a Diar object
        """

        # print(self.transition_probabilities)
        # print(self.observation)

        path = numpy.ones((self.nb_features, self.nb_clusters), 'int32') * -1
        path[0, :] = numpy.arange(self.nb_clusters)
        out_diarization = Diar()

        for row in table:
            start = row['start']
            stop = min(row['stop'], self.nb_features-1)
            logging.debug('perform from %d to %d', start, stop)

            for t in range(start, stop+1):
                tmp = self.observation[t - 1, :] + self.transition_probabilities
                self.observation[t, :] += numpy.max(tmp, axis=1)
                path[t, :] = numpy.argmax(tmp, axis=1)

            max_pos = numpy.argmax(self.observation[stop, :])
            out_diarization.append(show=self.show, start=stop - 1, stop=stop,
                             cluster=self.cluster_list[max_pos])
            for t in range(stop - 1, start, -1):
                max_pos = path[t, max_pos]
                cluster = self.cluster_list[max_pos]
                if (out_diarization[-1]['start'] == t) and (
                    out_diarization[-1]['cluster'] == cluster):
                    out_diarization[-1]['start'] -= 1
                else:
                    out_diarization.append(show=self.show, start=t - 1, stop=t,
                                     cluster=cluster)
        out_diarization.sort()
        # self.observation = None
        return out_diarization


def viterbi_decoding(cep, diarization, penalty):
    init_diarization = copy.deepcopy(diarization)
    if len(init_diarization) <=1:
        return init_diarization
    for seg in init_diarization:
        seg['cluster'] = 'init'
    init_diarization.pack()
    hmm = Viterbi(cep, diarization, exit_penalties=[penalty])
    hmm.train()
    hmm.emission()
    return hmm.decode(init_diarization)


class ViterbiMap(Viterbi):
    eps = 1.0e-200

    def __init__(self, featureServer, diarization, ubm, exit_penalties=[0],
                 loop_penalties=[0], alpha=0.9, linear=False):
        assert isinstance(featureServer, FeaturesServer), 'First parameter should be a FeatureServer'

        self.featureServer = featureServer
        self.ubm = ubm
        self.alpha = alpha
        self.linear = linear
        self.diarization = diarization
        shows = self.diarization.unique('show')
        self.cluster_list = self.diarization.unique('cluster')
        self.nb_clusters = len(self.cluster_list)
        if len(shows) > 1:
            raise Exception('diarization indexes serveal shows')
        self.cep, lbl = self.featureServer.load(shows[0])

        self.nb_features = self.cep.shape[0]
        self.show = shows.pop()
        self.mixtures = None
        self.names = None
        self.observation = None
        self.observation_ubm = None
        self.transition_probabilities = None
        self.exit_penalties = exit_penalties
        self.loop_penalties = loop_penalties

    def train(self):
            idmap = self.diarization.id_map()
            stat=StatServer(idmap, self.ubm)
            stat.accumulate_stat(ubm=self.ubm, feature_server=self.featureServer, seg_indices=range(stat.segset.shape[0]), num_thread=1)
            stat = stat.sum_stat_per_model()[0]
            self.mixtures = stat.adapt_mean_MAP(self.ubm, self.alpha, linear=self.linear)
            self.names = self.mixtures.modelset
            #print(self.names)
            #print(self.mixtures.stat1[:, 0:24])
            self._init_transition()

    def emission(self, ubm=False):
        self.observation = numpy.zeros((self.nb_features, self.nb_clusters))
        self.observation_ubm = None
        if ubm:
            self.observation_ubm = numpy.zeros((self.nb_features, 1))
            lp = self.ubm.compute_log_posterior_probabilities(self.cep)
            #self.observation_ubm = numpy.log(numpy.sum(numpy.exp(lp), axis=1))
            pp_max = numpy.max(lp, axis=1)
            self.observation_ubm = pp_max + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).transpose()), axis=1))

        for i in range(0, self.nb_clusters):
            logging.info('emission name: %s', self.mixtures.modelset[i])
            mean = self.mixtures.stat1[i, :]
            lp = self.ubm.compute_log_posterior_probabilities(self.cep, mean)
            #self.observation[:, i] = numpy.log(numpy.sum(numpy.exp(lp), axis=1))
            pp_max = numpy.max(lp, axis=1)
            self.observation[:, i] = pp_max + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).transpose()), axis=1))
            #print(self.observation[0:10, i])
