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

import sys
import os
from sidekit.features_extractor import FeaturesExtractor
from sidekit.features_server import FeaturesServer
import logging
import re
import numpy

def str2str_normalize(name):
    """
    removes accents and replace '_' by '_' the the string speaker
    :param name: the string to nomalize
    :return:
    """
    name = name.translate(str.maketrans('ÀÁÂÃÄÅàáâãäåÒÓÔÕÖØòóôõöøÈÉÊËèéêëÇçÌÍÎÏìíîïÙÚÛÜùúûüÿÑñ','AAAAAAaaaaaaOOOOOOooooooEEEEeeeeCcIIIIiiiiUUUUuuuuyNn')).lower()
    name = name.translate(str.maketrans("'",'_'))

    name = name.translate(str.maketrans('-','_'))
    return re.sub('_+','_',name)


def path_show_ext(fullpath, shortext=False):
    """
    splits a full file path into path, basename and extension
    :param fullpath: str
    :return: the path, the basename and the extension
    """
    tmp = os.path.splitext(fullpath)
    ext = tmp[1]
    p = tmp[0]
    if shortext == False:
        while tmp[1] != '':
            tmp = os.path.splitext(p)
            ext = tmp[1] + ext
            p = tmp[0]

    path = os.path.dirname(p)
    if path == '':
        path = '.'
    base = os.path.basename(p)
    return path, base, ext

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                             new_distances[-1])))
        distances = new_distances
    return distances[-1]


def hms(s):
    """
    conversion of seconds into hours, minutes and secondes
    :param s:
    :return: int, int, float
    """
    h = int(s) // 3600
    s %= 3600
    m = int(s) // 60
    s %= 60
    return '{:d}:{:d}:{:.2f}'.format(h, m, s)


def get_feature_extractor(audio_filename_structure, type_feature_extractor):
    if type_feature_extractor == 'sid':
        fe = FeaturesExtractor(audio_filename_structure=audio_filename_structure,
             feature_filename_structure=None,
             sampling_frequency=16000,
             lower_frequency=133.3333,
             higher_frequency=6855.4976,
             filter_bank="log",
             filter_bank_size=40,
             window_size=0.025,
             shift=0.01,
             ceps_number=13,
             pre_emphasis=0.97,
             keep_all_features=True,
             vad='percentil',
             #vad=None,
             save_param=["energy", "cep", "vad"]
            )
    elif type_feature_extractor == 'sid8k':
        fe = FeaturesExtractor(audio_filename_structure=audio_filename_structure,
             feature_filename_structure=None,
             sampling_frequency=8000,
             lower_frequency=0,
             higher_frequency=4000,
             filter_bank="log",
             filter_bank_size=24,
             window_size=0.025,
             shift=0.01,
             ceps_number=12,
             pre_emphasis=0.95,
             keep_all_features=True,
             #vad='percentil',
             vad=None,
             save_param=["energy", "cep", "vad"]
            )
    elif type_feature_extractor == '8k' or type_feature_extractor == '8kcms'\
            or type_feature_extractor == '8ksns':
        fe = FeaturesExtractor(audio_filename_structure=audio_filename_structure,
             feature_filename_structure=None,
             sampling_frequency=8000,
             lower_frequency=0,
             higher_frequency=4000,
             filter_bank="log",
             filter_bank_size=24,
             window_size=0.025,
             shift=0.01,
             ceps_number=13,
             pre_emphasis=0.97,
             keep_all_features=True,
             #vad='percentil',
             vad=None,
             save_param=["energy", "cep", "vad"]
            )
    elif type_feature_extractor == 'basic':
        fe = FeaturesExtractor(audio_filename_structure=audio_filename_structure,
             feature_filename_structure=None,
             sampling_frequency=16000,
             lower_frequency=133.3333,
             higher_frequency=6855.4976,
             filter_bank="log",
             filter_bank_size=40,
             window_size=0.025,
             shift=0.01,
             ceps_number=13,
             pre_emphasis=0.97,
             keep_all_features=True,
             vad=None,
             save_param=["energy", "cep", "vad"]
            )
    else:
        logging.error('in get_feature_server, type_fe not found: ' + type_feature_extractor)
        return None
    return fe


def get_feature_server(filename_structure, feature_server_type):
    path, show, ext = path_show_ext(filename_structure)
    feature_filename_structure = None
    logging.info(path+' ## '+show+' ## '+ext)
    if ext.endswith('.h5') or ext.endswith('.hdf5'):
        feature_extractor = None
        feature_filename_structure = filename_structure
        logging.info('feature extractor --> None')
    else:
        audio_filename_structure = filename_structure
        feature_extractor = get_feature_extractor(audio_filename_structure, type_feature_extractor=feature_server_type)
    logging.info('-'*20)
    logging.info(feature_extractor)
    logging.info('-'*20)
    if feature_server_type == 'basic':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('energy', 'cep'),
                 keep_all_features=True)
    elif feature_server_type == 'sns':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('cep'),
                 delta=True,
                 keep_all_features=True)
    elif feature_server_type == 'sns_dnn':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('cep'),
                 delta=True,
                 context=(31, 31),
                 keep_all_features=True)
    elif feature_server_type == 'sid':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('energy', 'cep'),
                 feat_norm='cmvn_sliding',
                 delta=True,
                 double_delta=True,
                 keep_all_features=True)
    elif feature_server_type == 'sid8k':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('cep'),
                 feat_norm='cmvn_sliding',
                 delta=True,
                 double_delta=False,
                 keep_all_features=True)
    elif feature_server_type == '8k':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('cep'),
                 #delta=True,
                 keep_all_features=True)
    elif feature_server_type == '8ksns':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('cep'),
                 delta=True,
                 keep_all_features=True)
    elif feature_server_type == '8kcms':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('cep'),
                 feat_norm='cms',
                 #delta=True,
                 keep_all_features=True)
    elif feature_server_type == 'vad':
        feature_server = FeaturesServer(features_extractor=feature_extractor,
                 feature_filename_structure=feature_filename_structure,
                 dataset_list=('energy'),
                 keep_all_features=True)
    else:
        logging.error('in get_feature_server, feature_server_type not found: ' + feature_server_type)
        return None
    logging.info(feature_server)
    return feature_server

# def get_feature_server(input_dir='./{s}.h5', feature_server_type):
#     logging.info('get_feature_server type: '+feature_server_type)
#     if feature_server_type == 'diarization':
#         return FeaturesServer_test(input_dir=input_dir,
#                     config='diar_16k')
#     elif feature_server_type == 'sid':
#         return FeaturesServer_test(input_dir=input_dir,
#                     config='diar_16k', log_e=True, delta=True,
#                     double_delta=True, feat_norm='cms_sliding')
#     elif feature_server_type == 'sad':
#         return FeaturesServer_test(input_dir=input_dir,
#                     config='diar_16k', log_e=False, delta=True, double_delta=False)
#     else:
#         logging.error('in get_feature_server, feature_server_type not found: ' + feature_server_type)
#         return None


# def save_mfcc(diarization, audio_dir, mfcc_fn, feature_server_type):
#     fh = h5py.File(mfcc_fn, "w")
#     diar_out = diarization.copy_structure()
#     shows = diarization.make_index(['show'])
#
#     for show in shows:
#         # logging.info('mfcc: '+ show)
#         show_diar = shows[show]
#         model_iv = ModelIV()
#         feature_server = get_feature_server(audio_dir, feature_server_type=feature_server_type)
#         model_iv.set_feature_server(feature_server)
#         model_iv.set_diar(show_diar)
#         if feature_server_type == 'sid':
#             model_iv.vad()
#         else:
#             model_iv.diar_vad = show_diar
#
#         cep_full, _ = feature_server.load(show)
#         cluster_list = model_iv.diar_vad.make_index(['cluster'])
#         index = model_iv.diar_vad.features_by_cluster(show=show, cep_len=cep_full.shape[0])
#         for cluster in cluster_list:
#             logging.info('mfcc: '+show+' '+cluster)
#             mfcc_fn = show+'/'+cluster
#             cep = cep_full[index[cluster], :]
#             vad = numpy.ones(cep.shape[0])
#             diar_out.append(show=mfcc_fn, start=0, stop=cep.shape[0], cluster=cluster)
#             logging.info(cep.shape)
#             write_hdf5(mfcc_fn, fh, cep, None, None, None, label=vad)
#     return diar_out

# def save_mfcc(diarization, audio_dir='./', mfcc_fn='./out.h5', feature_server_type='sid'):
#     fh = h5py.File(mfcc_fn, "w")
#     shows = diarization.unique('show')
#     diar_out = diarization.copy_structure()
#     shows = diarization.make_index(['show'])
#     for show in shows:
#         # logging.info('mfcc: '+ show)
#         show_diar = shows[show]
#         model_iv = ModelIV()
#         feature_server = get_feature_server(audio_dir, feature_server_type=feature_server_type)
#         model_iv.set_feature_server(feature_server)
#         model_iv.set_diar(show_diar)
#         if feature_server_type == 'sid':
#             model_iv.vad()
#         else:
#             model_iv.diar_vad = show_diar
#         feature_server.load(show)
#         cep_full = feature_server.cep[0]
#         cluster_list = model_iv.diar_vad.make_index(['cluster'])
#         index = model_iv.diar_vad.features_by_cluster(show=show, cep_len=cep_full.shape[0])
#         for cluster in cluster_list:
#             logging.info('mfcc: '+show+' '+cluster)
#             mfcc_fn = show+'/'+cluster
#             cep = cep_full[index[cluster], :]
#             vad = numpy.ones(cep.shape[0])
#             diar_out.append(show=mfcc_fn, start=0, stop=cep.shape[0], cluster=cluster)
#             write_hdf5(mfcc_fn, fh, cep, label=vad)
#     return diar_out

class FeatureServerFake(FeaturesServer):
    def __init__(self, cep):
        self.cep = cep

    def load(self, show, channel=0, input_feature_filename=None, label=None, start=None, stop=None):
        return self.cep, numpy.ones(self.cep.shape[0], dtype='bool')

class FeatureServerCache(FeaturesServer):
    def __init__(self, featuresServer):
        self.shows = dict()
        self.featuresServer = featuresServer


    def load(self, show, channel=0, input_feature_filename=None, label=None, start=None, stop=None):
        key = show
        if label is not None:
            key += '##'+label
        #if start is not None:
        #    key += '##'+str(start)+'##'+str(stop)
        #for k in self.shows:
        #    logging.info('key: %s', k)
        if key in self.shows:
            #logging.info('load from mem '+key)
            cep = self.shows[key][start:stop,:]
            return cep, numpy.ones(cep.shape[0], dtype='bool')
        else:
            #logging.info('load from disque %s', key)
            cep, lbl = self.featuresServer.load(show, label=label, start=start, stop=stop)
            self.shows[key] = cep
            #logging.info('add: %s %d %d', key, (key in self.shows), True)
            return cep, lbl
