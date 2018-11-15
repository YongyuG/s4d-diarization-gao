# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2016 Sylvain Meignier and Anthony Larcher

    :mod:`features_server` provides methods to manage features

"""
import os
import multiprocessing
import logging
from sidekit import PARALLEL_MODULE
from sidekit.frontend.features import *
from sidekit.frontend.vad import *
from sidekit.frontend.io import *
from sidekit.frontend.normfeat import *
from sidekit.sidekit_wrappers import *
import sys
import numpy as np
import ctypes
from sidekit.features_server import FeaturesServer
if sys.version_info.major == 3:
    import queue as Queue
else:
    import Queue
# import memory_profiler


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


class FeaturesServer_test(FeaturesServer):
    """
    A class for acoustic feature management.
    FeaturesServer should be used to extract acoustic features (MFCC or LFCC)
    from audio files in SPHERE, WAV or RAW PCM format.
    It can also be used to read and write acoustic features from and to disk
    in SPRO4 or HTK format.

    :attr input_dir: directory where to load audio or feature files
    :attr input_file_extension: extension of the incoming files
    :attrlabel_dir: directory where to read and write label files
    :attr label_files_extension: extension of label files to read and write
    :attr from_file: format of the input files to read, can be `audio`, `spro4`
        or `htk`, for audio files, format is given by the extension
    :attr config: pre-defined configuration for speaker diarization or recognition
        in 8 or 16kHz. Default is speaker recognition 8kHz
    :attr single_channel_extension: list with a single extension to add to 
        the audio filename when processing a single channel file. 
        Default is empty, means the feature file has the same speaker as
        the audio file
    :attr double_channel_extension: list of two channel extension to add 
        to the audio filename when processing two channel files. 
        Default is ['_a', '_b']
    :attr sampling_frequency: sample frequency in Hz, default is None, 
        determine when reading the audio file
    :attr lower_frequency: lower frequency limit of the filter bank
    :attr higher_frequency: higher frequency limit of the filter bank
    :attr linear_filters: number of linear filters to use for LFCC extraction
    :attr log_filters: number of linear filters to use for MFCC extraction
    :attr window_size: size of the sliding window in seconds
    :attr shift: time shift between two feature vectors
    :attr ceps_number: number of cepstral coefficients to extract
    :attr snr: snr level to consider for SNR-based voice activity detection
    :attr vad: type of voice activity detection to use, can be 'snr', 'energy' 
        (using a three Gaussian detector) or 'label' when reading the info from 
        pre-computed label files
    :attr feat_norm: normalization of the acoustic features, can be 
        'cms' for cepstral mean subtraction, 'mvn' for mean variance 
        normalization or 'stg' for short term Gaussianization
    :attr log_e: boolean, keep log energy
    :attr delta: boolean, add the first derivative of the cepstral coefficients
    :attr double_delta: boolean, add the second derivative of the cepstral 
        coefficients
    :attr rasta: boolean, perform RASTA filtering
    :attr keep_all_features: boolean, if False, only features labeled as 
        "speech" by the vad are saved if True, all features are saved and 
        a label file is produced

    """

    def __init__(self, input_dir=None,
                 feature_id=None,
                 config=None,
                 sampling_frequency=None,
                 lower_frequency=None,
                 higher_frequency=None,
                 linear_filters=None,
                 log_filters=None,
                 window_size=None,
                 shift=None,
                 ceps_number=None,
                 snr=None,
                 vad=None,
                 feat_norm=None,
                 log_e=None,
                 dct_pca=False,
                 dct_pca_config=None,
                 sdc=False,
                 sdc_config=None,
                 delta=None,
                 double_delta=None,
                 delta_filter=None,
                 rasta=None,
                 keep_all_features=None,
                 spec=False,
                 mspec=False,
                 mask=None
                 ):
        """ Process of extracting the feature frames (LFCC or MFCC) from an audio signal.
        Speech Activity Detection, MFCC (or LFCC) extraction and normalization.
        Can include RASTA filtering, Short Term Gaussianization, MVN and delta
        computation.

        :param input_dir: directory where to find the audio files.
                Default is ./
        :param input_file_extension: extension of the audio files to read.
                Default is 'sph'.
        :param label_dir: directory where to store label files is required.
                Default is ./
        :param label_file_extension: extension of the label files to create.
                Default is '.lbl'.
        :param configuration file : 'diar_16k', 'sid_16k', 'diar_8k' or 'sid_8k'
        """

        self.input_dir = './'
        self.from_file = 'audio'
        self.feature_id = 'ceps'
        self.sampling_frequency = 8000
        self.lower_frequency = 0
        self.higher_frequency = self.sampling_frequency / 2.
        self.linear_filters = 0
        self.log_filters = 40
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 13
        self.snr = 40
        self.vad = None
        self.feat_norm = None
        self.log_e = False
        self.dct_pca = False        
        self.dct_pca_config = (12, 12, None)
        self.sdc = False
        self.sdc_config = (1, 3, 7)
        self.delta = False
        self.double_delta = False
        self.delta_filter = np.array([.25, .5, .25, 0, -.25, -.5, -.25])
        self.mask = None
        self.rasta = False
        self.keep_all_features = False
        self.spec = False
        self.mspec = False
        self.single_channel_extension = ['']
        self.double_channel_extension = ['_a', '_b']

        # If a predefined config is chosen, apply it
        if config == 'diar_16k':
            self._config_diar_16k()
        elif config == 'diar_8k':
            self._config_diar_8k()
        elif config == 'sid_8k':
            self._config_sid_8k()
        elif config == 'sid_16k':
            self._config_sid_16k()
        elif config == 'fb_8k':
            self._config_fb_8k()
        elif config is None:
            pass
        else:
            raise Exception('unknown configuration value')


        # Manually entered parameters are applied
        if input_dir is not None:
            self.input_dir = input_dir
        if feature_id is not None:
            self.feature_id = feature_id
        if sampling_frequency is not None:
            self.sampling_frequency = sampling_frequency
        if lower_frequency is not None:
            self.lower_frequency = lower_frequency
        if higher_frequency is not None:
            self.higher_frequency = higher_frequency
        if linear_filters is not None:
            self.linear_filters = linear_filters
        if log_filters is not None:
            self.log_filters = log_filters
        if window_size is not None:
            self.window_size = window_size
        if shift is not None:
            self.shift = shift
        if ceps_number is not None:
            self.ceps_number = ceps_number
        if snr is not None:
            self.snr = snr
        if vad is not None:
            self.vad = vad
        if feat_norm is not None:
            self.feat_norm = feat_norm
        if log_e is not None:
            self.log_e = log_e
        if dct_pca is not None:
            self.dct_pca = dct_pca
        if dct_pca_config is not None:
            self.dct_pca_config = dct_pca_config
        if sdc is not None:
            self.sdc = sdc
        if sdc_config is not None:
            self.sdc_config = sdc_config
        if delta is not None:
            self.delta = delta
        if double_delta is not None:
            self.double_delta = double_delta
        if delta_filter is not None:
            self.delta_filter = delta_filter
        if mask is not None:
            self.mask = mask
        if rasta is not None:
            self.rasta = rasta
        if keep_all_features is not None:
            self.keep_all_features = keep_all_features
        if spec:
            self.spec = True
        if mspec:
            self.mspec = True
        
        self.cep = []
        self.label = []
        self.show = 'empty'
        self.audio_filename = 'empty'
        root, ext = os.path.splitext(self.input_dir)
        if ext == '.hdf5' or ext == '.h5':
            self.from_file = 'hdf5'

    def __repr__(self):
        ch = '\t show: {} keep_all_features: {} from_file: {}\n'.format(
            self.show, self.keep_all_features, self.from_file)
        ch += '\t inputDir: {}  \n'.format(self.input_dir)
        ch += '\t lower_frequency: {}  higher_frequency: {} \n'.format(
            self.lower_frequency, self.higher_frequency)
        ch += '\t sampling_frequency: {} '.format(self.sampling_frequency)
        ch += '\t linear_filters: {}  or log_filters: {} \n'.format(
            self.linear_filters, self.log_filters)
        ch += '\t ceps_number: {}  window_size: {} shift: {} \n'.format(
            self.ceps_number, self.window_size, self.shift)
        ch += '\t vad: {}  snr: {} \n'.format(self.vad, self.snr)
        ch += '\t feat_norm: {} rasta: {} \n'.format(self.feat_norm, self.rasta)
        ch += '\t log_e: {} delta: {} double_delta: {} \n'.format(self.log_e,
                                                                  self.delta,
                                                                  self.double_delta)
        return ch

    def _config_diar_16k(self):
        """
        12 MFCC + E, no normalization
        """
        self.sampling_frequency = 16000
        self.lower_frequency = 133.3333
        self.higher_frequency = 6855.4976
        self.linear_filters = 0
        self.log_filters = 40
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 13
        self.snr = 40
        self.vad = None
        self.feat_norm = None
        self.log_e = True
        self.delta = False
        self.double_delta = False
        self.rasta = False
        self.keep_all_features = True

    def _config_diar_8k(self):
        """
        12 MFCC + E, no normalization
        """
        self.sampling_frequency = 8000
        self.lower_frequency = None
        self.higher_frequency = None
        self.linear_filters = 0
        self.log_filters = 24
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 13
        self.snr = 40
        self.vad = None
        self.feat_norm = None
        self.log_e = True
        self.delta = False
        self.double_delta = False
        self.rasta = False
        self.keep_all_features = True

    def _config_sid_16k(self):
        """
        19 MFCC + E + D + DD, normalization cmvn
        """
        self.sampling_frequency = 16000
        self.lower_frequency = 133.3333
        self.higher_frequency = 6855.4976
        self.linear_filters = 0
        self.log_filters = 40
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 13
        self.snr = 40
        self.vad = 'snr'
        self.feat_norm = 'cmvn'
        self.log_e = True
        self.delta = True
        self.double_delta = True
        self.rasta = True
        self.keep_all_features = False

    def _config_sid_8k(self):
        """
        19 MFCC + E + D + DD, normalization cmvn
        """
        self.sampling_frequency = 8000
        self.lower_frequency = 200
        self.higher_frequency = 3800
        self.linear_filters = 0
        self.log_filters = 24
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 13
        self.snr = 40
        self.vad = 'snr'
        self.feat_norm = 'cmvn'
        self.log_e = True
        self.delta = True
        self.double_delta = True
        self.rasta = True
        self.keep_all_features = False

    def _config_fb_8k(self):
        """
        19 MFCC + E + D + DD, normalization cmvn
        """
        self.sampling_frequency = 8000
        self.lower_frequency = 300
        self.higher_frequency = 3400
        self.linear_filters = 0
        self.log_filters = 40
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 0
        self.snr = 40
        self.vad = None
        self.feat_norm = None
        self.log_e = False
        self.delta = False
        self.double_delta = False
        self.rasta = False
        self.keep_all_features = True
        self.mspec = True
 
    def _config_lid_8k_sdc(self):
        """
        7 MFCC + 1 - 3 - 7 SDC
        """
        self.sampling_frequency = 8000
        self.lower_frequency = 300
        self.higher_frequency = 3400
        self.linear_filters = 0
        self.log_filters = 24
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 7
        self.snr = 40
        self.vad = 'snr'
        self.feat_norm = None
        self.log_e = False
        self.delta = False
        self.double_delta = False
        self.sdc = True
        self.sdc_config = (1, 3, 7)
        self.rasta = False
        self.keep_all_features = False

    def _features(self, show):
        cep = None
        label = None
        window_sample = int(self.window_size * self.sampling_frequency)
        shift_sample = int(self.shift * self.sampling_frequency)

        audio_filename = self.input_dir.format(s=show)
        logging.debug('--> '+audio_filename)
        if not os.path.isfile(audio_filename):
            logging.error('%s %s', audio_filename, show)
            raise IOError('File ' + audio_filename + ' not found')
        logging.info('read audio')
        x, rate = read_audio(audio_filename, self.sampling_frequency)

        if rate != self.sampling_frequency:
            raise "file rate don't match the rate of the feature server configuration"
        self.audio_filename = audio_filename
        logging.info('size of signal: %f len %d type size %d', x.nbytes/1024/1024, len(x), x.nbytes/len(x))

        if x.ndim == 1:
            x = x[:, np.newaxis]

        for i in range(0, 200, 5):
            print('==> ', i, x[i:i+5])
        
        channel_ext = []
        channel_nb = x.shape[1]
        np.random.seed(0)
        #x[:, 0] += 0.0001 * numpy.random.randn(x.shape[0])

        if channel_nb == 1:
            channel_ext.append('')
            # Random noise is added to the input signal to avoid zero frames.
        elif channel_nb == 2:
            channel_ext.append('_a')
            channel_ext.append('_b')
            #x[:, 1] += 0.0001 * numpy.random.randn(x.shape[0])

        # Process channels one by one
        for chan, chan_ext in enumerate(channel_ext):
            l = x.shape[0]

            dec = shift_sample * 250 * 25000 + window_sample
            dec2 = window_sample - shift_sample
            start = 0
            end = min(dec, l)
            while start < l - dec2:
                # if end < l:
                logging.info('process part : %f %f %f',
                             start / self.sampling_frequency,
                             end / self.sampling_frequency,
                             l / self.sampling_frequency)

                tmp = self._features_chan(show, channel_ext, x[start:end, chan])

                if cep is None:
                    cep = []
                    label = []
                    cep.append(tmp[0])
                    label.append(tmp[1])
                else:
                    cep.append(tmp[0])
                    label.append(tmp[1])
                start = end - dec2
                end = min(end + dec, l)
                if cep[-1].shape[0] > 0:
                    logging.info('!! size of signal cep: %f len %d type size %d', cep[-1].nbytes/1024/1024, len(cep[-1]),
                             cep[-1].nbytes/len(cep[-1]))
        del x
        # Smooth the cluster_list and fuse the channels if more than one.
        logging.info('Smooth the cluster_list and fuse the channels if more than one')
        if self.vad is not None:
            label = label_fusion(label)
        self._normalize(label, cep)

        # Keep only the required features and save the appropriate files
        # which are either feature files alone or feature and label files
        if not self.keep_all_features:
            logging.info('no keep all')
            for chan, chan_ext in enumerate(channel_ext):
                cep[chan] = cep[chan][label[chan]]
                label[chan] = label[chan][label[chan]]

        return cep, label

    def _features_chan(self, show, channel_ext, x):

        """Compelete the overwhole process of extracting the feature frames
        (LFCC or MFCC) from an audio signal.
        Speech Activity Detection, MFCC (or LFCC) extraction and normalization.
        Can include RASTA filtering, Short Term Gaussianization, MVN and delta
        computation.

        :param show: speaker of the file.
        """
        # If the size of the signal is not enough for one frame, return zero features
        if x.shape[0] < self.sampling_frequency * self.window_size:
            cep_size = self.ceps_number * (1 + int(self.delta) + int(self.double_delta))\
                       + int(self.mspec) * (self.linear_filters + self.log_filters)
            cep = np.empty((0, cep_size))
            label = np.empty((0, 1))

        # Extract cepstral coefficients
        else:
            c = mfcc(x, fs=self.sampling_frequency,
                 lowfreq=self.lower_frequency,
                 maxfreq=self.higher_frequency,
                 nlinfilt=self.linear_filters,
                 nwin=self.window_size, nlogfilt=self.log_filters,
                 nceps=self.ceps_number, get_spec=self.spec, 
                 get_mspec=self.mspec)
            print('test MFCC: cep', c[0][0:5,:])
            print('test MFCC: e', c[1][0:5])

            if self.ceps_number == 0 and self.mspec:
                cep = c[3]
                label = self._vad(c[1], x, channel_ext, show)

            else:
                label = self._vad(c[1], x, channel_ext, show)

                cep = self._log_e(c)
                cep, label = self._rasta(cep, label)
                if self.delta or self.double_delta:
                    cep = self._delta_and_2delta(cep)
                elif self.dct_pca:
                    cep = pca_dct(cep, self.dct_pca_config[0],
                                  self.dct_pca_config[1],
                                  self.dct_pca_config[2])
                elif self.sdc:
                    cep = shifted_delta_cepstral(cep, d=self.sdc_config[0],
                                                 P=self.sdc_config[1],
                                                 k=self.sdc_config[2])
        return cep, label

    def _log_e(self, c):
        """If required, add the log energy as last coefficient"""
        if self.log_e:
            logging.info('keep log_e')
            return np.hstack((c[1][:, np.newaxis], c[0]))
        else:
            logging.info('don\'t keep c0')
            return c[0]

    def _vad(self, logEnergy, x, channel_ext, show):
        """
        Apply Voice Activity Detection.
        :param x:
        :param channel:
        :param window_sample:
        :param channel_ext:
        :param show:
        :return:
        """
        label = None
        if self.vad is None:
            logging.info('no vad')
            label = np.array([True] * logEnergy.shape[0])
        elif self.vad == 'snr':
            logging.info('vad : snr')
            window_sample = int(self.window_size * self.sampling_frequency)
            label = vad_snr(x, self.snr, fs=self.sampling_frequency,
                            shift=self.shift, nwin=window_sample)
        elif self.vad == 'energy':
            logging.info('vad : energy')
            label = vad_energy(logEnergy, distribNb=3,
                               nbTrainIt=8, flooring=0.0001,
                               ceiling=1.5, alpha=0.1)
        else:
            logging.warning('Wrong VAD type')
        return label

    def _rasta(self, cep, label):
        """
        Performs RASTA filtering if required.
        The two first frames are copied from the third to keep
        the length consistent
        !!! if vad is None: label[] is empty

        :param channel: speaker of the channel
        :return:
        """
        if self.rasta:
            logging.info('perform RASTA %s', self.rasta)
            cep = rasta_filt(cep)
            cep[:2, :] = cep[2, :]
            label[:2] = label[2]
            
        return cep, label

    def _delta_and_2delta(self, cep):
        """
        Add deltas and double deltas.
        :param cep: a matrix of cepstral cefficients
        
        :return: the cepstral coefficient stacked with deltas and double deltas
        """
        if self.delta:
            logging.info('add delta')
            delta = compute_delta(cep, filt=self.delta_filter)
            cep = np.column_stack((cep, delta))
            if self.double_delta:
                logging.info('add delta delta')
                double_delta = compute_delta(delta, filt=self.delta_filter)
                cep = np.column_stack((cep, double_delta))
        return cep

    def _normalize(self, label, cep):
        """
        Normalize features in place

        :param label:
        :return:
        """
        # Perform feature normalization on the entire session.
        if self.feat_norm is None:
            logging.info('no normalization')
            pass
        elif self.feat_norm == 'cms':
            logging.info('cms normalization')
            for chan, c in enumerate(cep):
                cms(cep[chan], label[chan])
        elif self.feat_norm == 'cmvn':
            logging.info('cmvn normalization')
            for chan, c in enumerate(cep):
                cmvn(cep[chan], label[chan])
        elif self.feat_norm == 'stg':
            logging.info('stg normalization')
            for chan, c in enumerate(cep):
                stg(cep[chan], label=label[chan])
        elif self.feat_norm == 'cmvn_sliding':
            logging.info('sliding cmvn normalization')
            for chan, c in enumerate(cep):
                cep_sliding_norm(cep[chan], win=301, center=True, reduce=True)
        elif self.feat_norm == 'cms_sliding':
            logging.info('sliding cms normalization')
            for chan, c in enumerate(cep):
                cep_sliding_norm(cep[chan], win=301, center=True, reduce=False)
        else:
            logging.warning('Wrong feature normalisation type')

    def load(self, show, id=None):
        """
        Load a cep from audio or mfcc file. This method loads all channels
        available in the file.
        
        :param show: the speaker of the show to load
        
        :return: the cep array and the label array
        """
        # test if features is already computed
        if self.show == show:
            return self.cep, self.label
        self.show = show
        if self.from_file == 'audio':
            logging.debug('compute MFCC: ' + show)
            logging.debug(self.__repr__())
            self.cep, self.label = self._features(show)
        elif self.from_file == 'hdf5':
            logging.debug('load hdf5: ' + show)
            input_filename = self.input_dir.format(s=show)
            with h5py.File(input_filename, "r") as hdf5_input_fh:
                logging.debug('*** '+input_filename+' '+show)
                vad = True
                if self.vad is None:
                    vad = False
                cep, label = read_hdf5(hdf5_input_fh, show, feature_id=self.feature_id, label=vad)
                self.cep = [cep]
                if label is None:
                    self.label = [np.array([True] * self.cep[0].shape[0])]
                else:
                    self.label = [label]
        else:
            raise Exception('unknown from_file value')

        if self.mask is not None:
            self.cep[0] = self._mask(self.cep[0])

        if not self.keep_all_features:
            logging.debug('!!! no keep all feature !!!')
            self.cep[0] = self.cep[0][self.label[0]]
            self.label[0] = [np.array([True] * self.cep[0].shape[0])]

        return self.cep, self.label

    def _mask(self, cep):
        """
        keep only the MFCC index present in the filter list
        :param cep:
        :return: return the list of MFCC given by filter list
        """
        if len(self.mask) == 0:
            raise Exception('filter list is empty')
        logging.debug('applied mask')
        return cep[:, self.mask]

    def save(self, show, filename, mfcc_format, and_label=True):
        """
        Save the cep array in file
        
        :param show: the speaker of the show to save (loaded if need)
        :param filename: the file speaker of the mffc file or a list of 2 filenames
            for the case of double channel files
        :param mfcc_format: format of the mfcc file taken in values
            ['pickle', 'spro4', 'htk']
        :param and_label: boolean, if True save label files
        
        :raise: Exception if feature format is unknown
        """
        self.load(show)

        hdf5_ouput_fh = h5py.File(filename, "w")
        logging.debug('save hdf5: ' + show)
        #write_hdf5(show, fh, feat, feat_type='ceps', label=None )
        write_hdf5(show, hdf5_ouput_fh, self.cep[0], label=self.label[0])
        hdf5_ouput_fh.close()


    @process_parallel_lists
    def save_list(self, audio_file_list, feature_file_list, mfcc_format, feature_dir, 
                  feature_file_extension, and_label=False, numThread=1):
        """
        Function that takes a list of audio files and extract features
        
        :param audio_file_list: an array of string containing the speaker of the feature
            files to load
        :param feature_file_list: list of feature files to save, should correspond to the input audio_file_list
        :param mfcc_format: format of the feature files to save, could be spro4, htk, pickle
        :param feature_dir: directory where to save the feature files
        :param feature_file_extension: extension of the feature files to save
        :param and_label: boolean, if True save the label files
        :param numThread: number of parallel process to run
        """
        logging.info(self)
        for audio_file, feature_file in zip(audio_file_list, feature_file_list):
            cep_filename = os.path.join(feature_dir, feature_file + feature_file_extension)
            self.save(audio_file, cep_filename, mfcc_format, and_label)

    def dim(self):
        if self.show != 'empty':
            return self.cep[0].shape[1]
        dim = self.ceps_number
        if self.log_e:
            dim += 1
        if self.delta:
            dim *= 2
        if self.double_delta:
            dim *= 2
        logging.warning('cep dim computed using featureServer parameters')
        return dim

    def save_parallel(self, input_audio_list, output_feature_list, mfcc_format, feature_dir,
                      feature_file_extension, and_label=False, numThread=1):
        """
        Extract features from audio file using parallel computation
        
        :param input_audio_list: an array of string containing the speaker
            of the audio files to process
        :param output_feature_list: an array of string containing the 
            speaker of the features files to save
        :param mfcc_format: format of the output feature files, could be spro4, htk, pickle
        :param feature_dir: directory where to save the feature files
        :param feature_file_extension: extension of the feature files to save
        :param and_label: boolean, if True save the label files
        :param numThread: number of parallel process to run
        """
        # Split the features to process for multi-threading
        loa = np.array_split(input_audio_list, numThread)
        lof = np.array_split(output_feature_list, numThread)
    
        jobs = []
        multiprocessing.freeze_support()
        for idx, feat in enumerate(loa):
            p = multiprocessing.Process(target=self.save_list,
                                        args=(loa[idx], lof[idx], mfcc_format, feature_dir,
                                              feature_file_extension, and_label))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

    def _load_and_stack_worker(self, input_queue, output):
        """Load a list of feature files into a Queue object
        
        :param input_queue: a Queue object
        :param output: a list of Queue objects to fill
        """
        while True:
            next_task = input_queue.get()
            
            if next_task is None:
                # Poison pill means shutdown
                output.put(None)
                input_queue.task_done()
                break
            
            # check which channel to keep from the file
            if next_task.endswith(self.double_channel_extension[0]) and (self.from_file == 'audio'):
                next_task = next_task[:-len(self.double_channel_extension[0])]
                output.put(self.load(next_task)[0][0])
            if next_task.endswith(self.double_channel_extension[1]) and self.from_file == 'audio':
                next_task = next_task[:-len(self.double_channel_extension[1])]
                output.put(self.load(next_task)[0][1])
            else:
                cep = self.load(next_task)[0][0]
                output.put(cep)
            
            input_queue.task_done()

    def load_and_stack(self, fileList, numThread=1):
        """Load a list of feature files and stack them in a unique ndarray. 
        The list of files to load is splited in sublists processed in parallel
        
        :param fileList: a list of files to load
        :param numThread: numbe of thead (optional, default is 1)
        """
        queue_in = multiprocessing.JoinableQueue(maxsize=len(fileList)+numThread)
        queue_out = []
        
        # Start worker processes
        jobs = []
        for i in range(numThread):
            queue_out.append(multiprocessing.Queue())
            p = multiprocessing.Process(target=self._load_and_stack_worker, 
                                        args=(queue_in, queue_out[i]))
            jobs.append(p)
            p.start()
        
        # Submit tasks
        for task in fileList:
            queue_in.put(task)

        for task in range(numThread):
            queue_in.put(None)
        
        # Wait for all the tasks to finish
        queue_in.join()
                   
        output = []
        for q in queue_out:
            while True:
                data = q.get()
                if data is None:
                    break
                output.append(data)

        for p in jobs:
            p.join()
        all_cep = np.concatenate(output, axis=0)

        return all_cep

    def load_and_stack_threading(self, fileList, numThread=1):
        """Load a list of feature files and stack them in a unique ndarray. 
        The list of files to load is splited in sublists processed in parallel
        
        :param fileList: a list of files to load
        :param numThread: numbe of thead (optional, default is 1)
        """
        queue_in = multiprocessing.JoinableQueue(maxsize=len(fileList)+numThread)
        queue_out = []
        
        # Start worker processes
        jobs = []
        for i in range(numThread):
            queue_out.append(Queue.Queue())
            p = threading.Thread(target=self._load_and_stack_worker, args=(queue_in, queue_out[i]))
            jobs.append(p)
            p.start()
        
        # Submit tasks
        for task in fileList:
            queue_in.put(task)

        for task in range(numThread):
            queue_in.put(None)
        
        # Wait for all the tasks to finish
        queue_in.join()
                   
        output = []
        for q in queue_out:
            while True:
                data = q.get()
                if data is None:
                    break
                output.append(data)

        for p in jobs:
            p.join()
        all_cep = np.concatenate(output, axis=0)

        return all_cep

    def mean_std(self, filename):
        feat = self.load(filename)[0][0]
        return feat.shape[0], feat.sum(axis=0), np.sum(feat**2, axis=0)
