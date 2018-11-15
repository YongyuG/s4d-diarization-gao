import copy
import numpy as np
import logging


def argmin(distances, nb):
    """
    Get argmin and min indexes between 0 and nb of a distance matrix
    :param distances: a numpy.ndarray
    :param nb: int
    :return: row and column indexes, the value
    """
    if nb <= 1:
        return 0, 0, np.inf
    tmp_dist = distances[0:nb, 0:nb]
    # numpy.nanargmin : give the absolute position in the matrix, ie 1 number
    # unravel_index: give the row and col positions
    try:
        i, j = np.unravel_index(np.nanargmin(tmp_dist), tmp_dist.shape)
    except ValueError:
        logging.warning('value are NaN, nb:'+str(nb))
        logging.warning(distances)
        logging.warning(tmp_dist)
        return 0, 0, np.inf

    v = distances[i, j]
    return i, j, v

def argmax(distances, nb):
    """
    Get argmin and min indexes between 0 and nb of a distance matrix
    :param distances: a numpy.ndarray
    :param nb: int
    :return: row and column indexes, the value
    """
    if nb <= 1:
        return 0, 0, np.inf
    tmp_dist = distances[0:nb, 0:nb]
    # numpy.nanargmin : give the absolute position in the matrix, ie 1 number
    # unravel_index: give the row and col positions
    try:
        i, j = np.unravel_index(np.nanargmax(tmp_dist), tmp_dist.shape)
    except ValueError:
        logging.warning('value are NaN, nb:'+str(nb))
        logging.warning(distances)
        logging.warning(tmp_dist)
        return 0, 0, -np.inf

    v = distances[i, j]
    return i, j, v


def roll(mat, j):
    """
    delete the line j and column j in the matrix
    :param mat: numpy.ndarray
    :param j: int
    :return: numpy.ndarray
    """
    return np.delete(np.delete(mat, j, 1), j, 0)


def bic_square_root(ni, nj, alpha, dim):
    """
    Compute a BIC square root distance described in [Stafylakis2010]_.

    .. [Stafylakis2010] T. Stafylakis, V. Katsouros, and G. Carayannis. The segmental bayesian information criterion and its applications to speaker diarization. Selected Topics in Signal Processing, IEEE Journal of, 4(5):857-866, 2010.

    :param ni: covariance matrix of speaker i
    :param nj: covariance matrix of speaker j
    :param alpha: a threshold
    :param dim: the dimenssion of the features
    :return: a float
    """
    nij = ni + nj
    constant_covariance = 0.5 * alpha * (0.5 * ((dim + 1) * dim))
    constant_mean = 0.5 * alpha * dim
    mean = ((np.sqrt(ni) * np.log(ni)) + (np.sqrt(nj) * np.log(nj))) - (np.sqrt(nij) * np.log(nij))
    covariance = (np.log(ni) + np.log(nj)) - np.log(nij)
    #print(ni, nj, nij, alpha, dim, constant_covariance, constant_mean, mean, covariance)
    return (constant_covariance * covariance) + (constant_mean * mean)

def stat_server_remove(stat_server, index):
    """
    " remove data at position index
    :param index: the index to remove
    """
    stat_server.segset = np.delete(stat_server.segset, index)
    stat_server.modelset = np.delete(stat_server.modelset, index)
    stat_server.start = np.delete(stat_server.start, index)
    stat_server.stop = np.delete(stat_server.stop, index)
    stat_server.stat0 = np.delete(stat_server.stat0, index, axis=0)
    stat_server.stat1 = np.delete(stat_server.stat1, index, axis=0)


def stat_server_merge(stat_server, i, j, wi=1.0, wj=1.0):
    """
    merge the ith and jth stat0 and stat1 into ith data, remove jth data
    :param i: index destination
    :param j: index removed
    """
    if stat_server.stop[i] != 0 and stat_server.stop[i] is not None:
        logging.warning('segment information will be wrong')
    stat_server.stat0[i, :] = (wi * stat_server.stat0[j, :] + wj * stat_server.stat0[i, :]) / (wi + wj)
    stat_server.stat1[i, :] = (wi * stat_server.stat1[j, :] + wj * stat_server.stat1[i, :]) / (wi + wj)
    stat_server_remove(stat_server, j)


def idmap_remove(idmap, index):
    """
    " remove data at position index
    :param index: the index to remove
    """
    idmap.leftids = np.delete(idmap.leftids, index)
    idmap.rightids = np.delete(idmap.rightids, index)
    idmap.start = np.delete(idmap.start, index)
    idmap.stop = np.delete(idmap.stop, index)


def scores_remove(scores, index_model=None, index_seg=None):
    """
    " remove data at position index_model and/or index_seg
    :param index_model: the index in model set to remove
    :param index_seg: the index in segment set to remove
    """

    if index_seg is not None:
        scores.segset = np.delete(scores.segset, index_seg)
        scores.scoremask = np.delete(scores.scoremask, index_seg, axis=1)
        scores.scoremat = np.delete(scores.scoremat, index_seg, axis=1)

    if index_model is not None:
        scores.modelset = np.delete(scores.modelset, index_model)
        scores.scoremask = np.delete(scores.scoremask, index_model, axis=0)
        scores.scoremat = np.delete(scores.scoremat, index_model, axis=0)


def scores2distance(scores, threshold):
    distance = (scores.scoremat + scores.scoremat.T) / 2.0 * -1.0
    np.fill_diagonal(distance, np.inf)
    min = np.min(distance)-1
    distance -= min
    np.fill_diagonal(distance, 0.0)
    t = -1.0 * threshold - min
    return distance, t

