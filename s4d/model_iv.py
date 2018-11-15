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

from sidekit import Mixture, StatServer, FactorAnalyser, Scores, Ndx, PLDA_scoring, cosine_scoring, mahalanobis_scoring, two_covariance_scoring
from sidekit.sidekit_io import *
import copy
import numpy as np

class ModelIV:
    def __init__(self, model_filename=None, nb_thread=1):
        self.ubm = None
        self.tv = None
        self.tv_mean = None
        self.tv_sigma = None
        self.sn_mean = None
        self.sn_cov = None
        self.plda_mean = None
        self.plda_f = None
        self.plda_g = None
        self.plda_sigma = None
        self.ivectors = None
        self.scores = None

        self.nb_thread = nb_thread
        self.model_filename = model_filename
        if model_filename is not None:
            self._load_model()

    def _load_model(self):
        print('load: ', self.model_filename)
        self.ubm = Mixture()
        self.ubm.read(self.model_filename, prefix='ubm/')
        self.tv, self.tv_mean, self.tv_sigma = read_tv_hdf5(self.model_filename)
        self.norm_mean, self.norm_cov = read_norm_hdf5(self.model_filename)

    def debug_model(self):
        print('ubm_mu: ', self.ubm.mu.shape)
        print('tv: ', self.tv.shape)
        print('tv mean: ', self.tv_mean.shape)
        print('tv Sigma: ', self.tv_sigma.shape)
        if self.plda_mean is not None:
            print('plda mean: ', self.plda_mean.shape)
            print('plda Sigma: ', self.plda_sigma.shape)
            print('plda F: ', self.plda_f.shape)
            print('plda G: ', self.plda_g.shape)
        if self.sn_mean is not None:
            print('sn_mean: ', self.sn_mean.shape)
            print('sn_cov: ', self.sn_cov.shape)

    def train(self, feature_server, idmap, normalization=True):
        stat = StatServer(idmap, distrib_nb=self.ubm.distrib_nb(), feature_size=self.ubm.dim()) 
        stat.accumulate_stat(ubm=self.ubm, feature_server=feature_server, seg_indices=range(stat.segset.shape[0]), num_thread=self.nb_thread)
        stat = stat.sum_stat_per_model()[0]
        
        fa = FactorAnalyser(mean=self.tv_mean, Sigma=self.tv_sigma, F=self.tv)
        self.ivectors = fa.extract_ivectors_single(self.ubm, stat)
        
        if normalization:
            self.ivectors.spectral_norm_stat1(self.norm_mean[:1], self.norm_cov[:1])

        return self.ivectors

    def score_cosine(self, use_wccn=True):
        wccn = None
        if use_wccn:
            wccn = read_key_hdf5(self.model_filename, 'wccn_choleski')
        ndx = Ndx(models=self.ivectors.modelset, testsegs=self.ivectors.modelset)
        self.scores = cosine_scoring(self.ivectors, self.ivectors, ndx, wccn=wccn, check_missing=False)

        return self.scores

    def score_mahalanobis(self, use_covariance=True):
        if use_covariance:
            m = read_key_hdf5(self.model_filename, 'mahalanobis_matrix')
        else:
            m = numpy.identity(self.tv.shape[2])
        ndx = Ndx(models=self.ivectors.modelset, testsegs=self.ivectors.modelset)
        self.scores = mahalanobis_scoring(self.ivectors, self.ivectors, ndx, m, check_missing=False)

        return self.scores

    def score_two_covariance(self):
        W = read_key_hdf5(self.model_filename, 'two_covariance/within_covariance')
        B = read_key_hdf5(self.model_filename, 'two_covariance/between_covariance')
        ndx = Ndx(models=self.ivectors.modelset, testsegs=self.ivectors.modelset)
        self.scores = two_covariance_scoring(self.ivectors, self.ivectors, ndx, W, B, check_missing=False)

        return self.scores

    def score_plda(self):
        self.plda_mean, self.plda_f, self.plda_g, self.plda_sigma = read_plda_hdf5(self.file_name)
        ndx = Ndx(models=self.ivectors.modelset, testsegs=self.ivectors.modelset)

        self.scores = PLDA_scoring(self.ivectors, self.ivectors, ndx, self.plda_mean, self.plda_f, self.plda_g, self.plda_sigma, p_known=0.0)

        return self.scores

#    def update(self, i, j):
#        cluster_list = self.diarization.make_index(['cluster'])
#
#        stat_server_merge(self.ivectors, i, j, 1, 1)
#        self.scores = self.score_plda()

    def score_plda_slow(self):
        self.plda_mean, self.plda_f, self.plda_g, self.plda_sigma = read_plda_hdf5(self.model_filename)
        local_ndx = Ndx(models=self.ivectors.modelset, testsegs=self.ivectors.modelset)

        enroll_copy = copy.deepcopy(self.ivectors)

        # Center the i-vectors around the PLDA mean
        enroll_copy.center_stat1(self.plda_mean)

        # Compute temporary matrices
        invSigma = np.linalg.inv(self.plda_sigma)
        I_iv = np.eye(self.plda_mean.shape[0], dtype='float')
        I_ch = np.eye(self.plda_g.shape[1], dtype='float')
        I_spk = np.eye(self.plda_f.shape[1], dtype='float')
        A = np.linalg.inv(self.plda_g.T.dot(invSigma).dot(self.plda_g) + I_ch)
        B = self.plda_f.T.dot(invSigma).dot(I_iv - self.plda_g.dot(A).dot(self.plda_g.T).dot(invSigma))
        K = B.dot(self.plda_f)
        K1 = np.linalg.inv(K + I_spk)
        K2 = np.linalg.inv(2 * K + I_spk)

        # Compute the Gaussian distribution constant
        alpha1 = np.linalg.slogdet(K1)[1]
        alpha2 = np.linalg.slogdet(K2)[1]
        constant = alpha2 / 2.0 - alpha1

        # Compute verification scores
        l = enroll_copy.segset.shape[0]
        scores = Scores()
        scores.scoremat = np.zeros((l, l))
        scores.modelset = enroll_copy.modelset
        scores.segset = enroll_copy.modelset
        scores.scoremask = local_ndx.trialmask

        # Project data in the space that maximizes the speaker separability
        enroll_tmp = B.dot(enroll_copy.stat1.T)

        # Compute verification scores
        # Loop on the models
        for model_idx in range(enroll_copy.modelset.shape[0]):

            s2 = enroll_tmp[:, model_idx].dot(K1).dot(enroll_tmp[:, model_idx])

            mod_plus_test_seg = enroll_tmp + np.atleast_2d(enroll_tmp[:, model_idx]).T

            tmp1 = enroll_tmp.T.dot(K1)
            tmp2 = mod_plus_test_seg.T.dot(K2)

            for seg_idx in range(model_idx, enroll_copy.segset.shape[0]):
                s1 = tmp1[seg_idx, :].dot(enroll_tmp[:, seg_idx])
                s3 = tmp2[seg_idx, :].dot(mod_plus_test_seg[:, seg_idx])
                scores.scoremat[model_idx, seg_idx] = (s3 - s1 - s2)/2. + constant
                scores.scoremat[seg_idx, model_idx] = (s3 - s1 - s2)/2. + constant
        self.scores = scores
        return scores

#def plda_scores_from_diar(model_fn, feature_server, diar, idmap, return_model=False):
#    model_iv = ModelIV(model_filename=model_fn, feature_server=feature_server, diarization=diar, idmap=idmap)

#    model_iv.train()
#    scores = model_iv.score_plda()
#    if return_model:
#        return scores, model_iv
#    return scores

#def cosine_scores_from_diar(model_fn, feature_server, diar, return_model=False):
#    model_iv = ModelIV(model_filename=model_fn, feature_server=feature_server, diarization=diar)
    #if vad:
    #model_iv.vad()

#    model_iv.train(normalization=False)
#    scores = model_iv.score_cosine()
#    if return_model:
#        return scores, model_iv
#    return scores