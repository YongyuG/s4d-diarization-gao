__author__ = 'meignier'

import logging

from s4d.diar import Diar
import numpy as np
import pandas as pd
from s4d.clustering.hac_bic import GaussFull
from s4d.clustering.hac_utils import bic_square_root
import copy
import scipy

def sanity_check(cep, show, cluster='init'):
    """
    Removes equal MFCC of *cep* and return a diarization.

    :param cep: numpy.ndarry containing MFCC
    :param show: speaker of the show
    :return: a dirization object
    """
    table = Diar()

    # 1- diff on cep[i] - cep[i+1]
    # 2- sum of the n MFCC
    # 3- take equal values, give a boolean array
    b = np.sum(np.diff(cep, axis=0), axis=1) == 0
    # make a xor on the boolean array, true index+1 correspond to a boundary
    bits = b[:-1] ^ b[1:]
    # convert true value into a list of feature indexes
    # append 0 at the beginning of the list, append the last index to the list
    idx = [0] + (np.arange(len(bits))[bits] + 1).tolist() + [cep.shape[0]]
    # for each pair of indexes (idx[i] and idx[i+1]), create a segment
    for i in range(0, len(idx) - 1, 2):
        table.append(show=show, start=idx[i], stop=idx[i + 1], cluster=cluster)

    return table


def init_seg(cep, show='empty', cluster='init'):
    """
    Return an initial segmentation composed of one segment from the first to the
    last feature in *cep*.

    :param cep: numpy.ndarry containing MFCC
    :param show: the speaker of the cep
    :param cluster: str
    :return: a Diar object
    """
    length = cep.shape[0]
    table_out = Diar()
    table_out.append(show=show, start=0, stop=length, cluster=cluster)
    return table_out


def adjust(cep, diarization):
    """
    Moves the border of segment of *diarization* into lowest energy region and split
    segments greater than 30s

    :todo: change numpy.convolve to the panada version

    :param cep: a numpy.ndarray containing MFCC
    :param diarization: a Diarization object
    :return: a Diar object
    """
    energy_index = 0
    box = np.ones(100) / 100

    smooth = np.convolve(cep[:, energy_index], box, mode='same')
    adj_table = _adjust(smooth, diarization)
    return _split_e(smooth, adj_table, 30*100)


def _adjust(smooth, diarization, window_size=25):
    """
    The segment boundaries of *diarization* are moved slightly: segment start and
    segment stop will be located in low energy regions.

    :param smooth: sliding means of the energy (numpy.ndarry)
    :param diarization: the diarization object to adjust
    :param window_size: the half size of the zone to find the minimum energy around a
    border
    :return: a Diar object
    """
    diarization_out = copy.deepcopy(diarization)
    diarization_out.sort(['start'])
    prev = diarization_out[0]
    for i in range(1, len(diarization_out)):
        cur = diarization_out[i]
        start = cur['start']
        p = np.argmin(smooth[start - window_size:start + window_size])
        l1 = p + start - window_size - prev['start']
        l2 = prev['stop'] - p + start - window_size
        if l1 > 500 and l2 > 500:
            prev['stop'] = p + start - window_size
            cur['start'] = p + start - window_size
        prev = cur
    return diarization_out


def _split_e(smooth, diarization, split_size):
    """
    Long segments of *diarization* are  cut recursively at their points of lowest
    energy in order to yield segments shorter than *split_size* seconds.

    :param smooth: sliding means of the energy (numpy.ndarray)
    :param diarization: a Diarization object
    :param split_size: maximum size of a segment
    :return: a Diar object
    """
    diarization_out = Diar()
    for segment in diarization:
        _split_seg(smooth, segment, 250, split_size, diarization_out.segments)
    return diarization_out


def _split_seg(smooth, segment, min_seg_size, split_size, lst):
    """
    *segment*, a long segment, is cut recursively at their points of lowest energy
    in order to yield segments shorter than *split_size* seconds. The new
    segments greater than *min_seg_size* are appended into *lst*

    :param smooth: sliding means of the energy (numpy.ndarry)
    :param segment: a segment
    :param min_seg_size: minimum size of a segment
    :param split_size: maximum size of a segment
    :param lst: the new segments are added to this list
    :return:
    """
    stop = segment['stop'] - min_seg_size
    start = segment['start'] + min_seg_size
    l = segment['stop'] - segment['start']
    if l > split_size:
        m = start + np.argmin(smooth[start:stop])
        row_left = copy.deepcopy(segment)
        row_left['stop'] = m
        row_right = copy.deepcopy(segment)
        row_right['start'] = m
        _split_seg(smooth, row_left, min_seg_size, split_size, lst)
        _split_seg(smooth, row_right, min_seg_size, split_size, lst)
    else:
        lst.append(copy.deepcopy(segment))



def div_gauss(cep, show='empty', win=250, shift=0):
    """
    Segmentation based on gaussian divergence.

    The segmentation detects the instantaneous change points corresponding to
    segment boundaries. The proposed algorithm is based on the detection of
    local maxima. It detects the change points through a gaussian divergence
    (see equation below), computed using Gaussians with diagonal covariance 
    matrices. The left and right gaussians are estimated over a five-second 
    window sliding along the whole signal (2.5 seconds for each gaussian, 
    given *win* =250 features).
    A change point, i.e. a segment boundary, is present in the middle of the
    window when the gaussian divergence score reaches a local maximum.


        :math:`GD(s_l,s_r)=(\\mu_r-\\mu_l)^t\\Sigma_l^{-1/2}\\Sigma_r^{-1/2}(\\mu_r-\\mu_l)`

    where :math:`s_l` is the left segment modeled by the mean :math:`\mu_l` and
    the diagonal covariance matrix :math:`\\Sigma_l`, :math:`s_r` is the right
    segment modeled by the mean :math:`\mu_r` and the diagonal covariance
    matrix :math:`\\Sigma_r`.

    :param cep: numpy array of frames
    :param show: speaker of the show
    :param win: windows size in number of frames
    :return: a diarization object (s4d annotation)
    """

    length = cep.shape[0]
    # start and stop of the rolling windows A
    start_a = win - 1  # end of NAN
    stop_a = length - win
    # start and stop of the rolling windows B
    start_b = win + win - 1  # end of nan + delay
    stop_b = length

    # put features in a Pandas DataFrame
    df = pd.DataFrame(cep)
    # compute rolling mean and std in the window of size win, get numpy array
    # mean and std have NAN at the beginning and the end of the output array
    #mean = pd.rolling_mean(df, win).values
    #std = pd.rolling_std(df, win).values
    r = df.rolling(window=win, center=False)
    mean = r.mean().values
    std = r.std().values

    # compute GD scores using 2 windows A and B
    dist = (np.square(mean[start_a:stop_a, :] - mean[start_b:stop_b, :]) / (
        std[start_a:stop_a, :] * std[start_b:stop_b, :])).sum(axis=1)

    # replace missing value to match cep size
    dist_pad = np.lib.pad(dist, (win - 1, win), 'constant',
                          constant_values=(dist[0], dist[-1]))

    # remove non-speech frame
    # find local maximal at + or - win size
    borders = scipy.signal.argrelmax(dist_pad, order=win)[0].tolist()
    # append the first and last
    borders = [0] + borders + [length]

    diarization_out = Diar()
    spk = 0
    for i in range(0, len(borders) - 1):
        diarization_out.append(show=show, start=shift+borders[i],
                         stop=shift+borders[i + 1], cluster='S' + str(spk))
        spk += 1
    return diarization_out


def segmentation(cep, diarization, win_size=250):
    diarization_out = Diar()
    for segment in diarization:
        l = segment.duration()
        # logging.info('start: ', seg['start'],'end: ', seg['stop'], 'len: ', l)
        if l > 2 * win_size:
            cep_seg = segment.seg_features(cep)
            tmp = div_gauss(cep_seg, show=segment['show'], win=win_size, shift=segment['start'])
            diarization_out.append_diar(tmp)
        else:
            diarization_out.append_seg(segment)

    i=0
    for segment in diarization_out:
        segment['cluster'] = 'S'+str(i)
        i += 1

    return diarization_out


def bic_linear(cep, diarization, alpha, sr=False):
    """
    This segmentation over the signal fuses consecutive segments of the same
    speaker from the start to the end of the record.  The measure employs
    the :math:`\Delta BIC` based on Bayesian Information Criterion , using full
    covariance Gaussians (see :class:`gauss.GaussFull`), as defined in equation below.

        :math:`\\Delta BIC_{i,j} = PBIC_{i+j} - PBIC_{i} - PBIC_{j} -  P`

        :math:`PBIC_{x}  = \\frac{n_x}{2} \\log|\\Sigma_x|`

        :math:`cst  = \\frac{1}{2} \\alpha \\left(d + \\frac{d(d+1)}{2}\\right)`

        :math:`P  = cst \\times log(n_i+n_j)`

    where :math:`|\\Sigma_i|`, :math:`|\\Sigma_j|` and :math:`|\\Sigma|` are the
    determinants of gaussians associated to the left and right segments
    :math:`i`, :math:`j`
    and :math:`i+j`. :math:`\\alpha` is a parameter to set up. The penalty
    factor :math:`P` depends on :math:`d`, the dimension of the cep, as
    well as on :math:`n_i` and :math:`n_j`, refering to the total length of
    left segment :math:`i` and right segment :math:`j` respectively.

    if *sr* is True, BIC distance is replaced by the square root bic
    (see :py:func:`clustering.hac_utils.bic_square_root`)

    :param cep: numpy.ndarray
    :param diarization: a Diarization object
    :param alpha: the threshold
    :param sr: boolean
    :return: a Diar object
    """
    # logger = logging.getLogger(__name__)

    diarization_out = copy.deepcopy(diarization)
    diarization_out.sort(['show', 'start'])
    dim = cep.shape[1]
    cst = GaussFull.cst_bic(dim, alpha)

    if len(diarization) <= 1:
        return diarization_out
    segment1 = diarization_out[0];
    features1 = segment1.seg_features(cep)
    model1 = GaussFull(segment1['cluster'], dim)
    model1.add(features1)
    model1.compute()
    i = 1

    while i < len(diarization_out):
        segment2 = diarization_out[i];
        if segment2['start'] > segment1['stop']+1:
            # logging.warning('there is a hole between segment')
            i += 1
            segment1 = segment2
            continue
        features2 = segment2.seg_features(cep)
        model2 = GaussFull(segment2['cluster'], dim)
        model2.add(features2)
        model2.compute()

        model12 = GaussFull.merge(model1, model2)
        p = cst * np.log(model1.count + model2.count)
        if sr:
            p = bic_square_root(model1.count, model2.count, alpha, dim)
        delta_bic = model12.partial_bic - model1.partial_bic - model2.partial_bic - p
        #print(i, v, p)
        if delta_bic < 0.0:
            logging.debug('linear remove %s %s: %i/%i %f', model1.name, model2.name, i,
                          len(diarization_out), delta_bic)
            segment1['stop'] = segment2['stop']
            model1 = model12
            del diarization_out[i]
        else:
            logging.debug('linear next %s %s: %i/%i %f', model1.name, model2.name, i,
                          len(diarization_out), delta_bic)
            segment1 = segment2
            model1 = model2
            i += 1
    return diarization_out

