import copy
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy as hac
import logging
from s4d.clustering.hac_utils import *


def information(merge, nb_merge, i, j, value):
    merge.append([nb_merge, i, j, value])


def hac_iv(diar, scores, method="complete", threshold=0.0):
    ldiar = copy.deepcopy(diar)
    lscores = copy.deepcopy(scores)
    # get the triangular part of the distances
    distances, t = scores2distance(lscores, threshold)

    #distance = numpy.copy((scores.scoremat + scores.scoremat.T) / 2.0) * -1.0
    #numpy.fill_diagonal(distance, numpy.inf)
    #min = numpy.min(distance)
    #distance -= min
    #numpy.fill_diagonal(distance, 0.0)
    distance_sym = squareform(distances)
    #t = -1.0 * threshold - min
    # cluster the data
    link = hac.linkage(distance_sym, method=method)
    # print(link)
    # assign new cluster
    # d : 'key' give the new names of cluster_list in values (a list)
    cluster_dict = dict()
    merge = list()
    i = 0
    cluster_list = scores.modelset.tolist()
    #print(cluster_list)
    while i < len(link) and link[i, 2] < t:
        # the cluster_list of the 2 clusters
        logging.debug('c0: {:d} c1: {:d} value: {:.4f}'.format(int(link[i, 0]),
                                                              int(link[i, 1]),
                                                              link[i, 2]))
        c0 = cluster_list[int(link[i, 0])]
        c1 = cluster_list[int(link[i, 1])]
        logging.debug(
            '\t c0: {} c1: {} value: {:.4f}'.format(c0, c1, link[i, 2]))
        information(merge, i, c0, c1, link[i, 2])
        if c1 in cluster_dict:
            # c0 is put in c1, and c1 is not empty
            cluster_dict[c1].append(c0)
        else:
            cluster_dict[c1] = [c0]
        if c0 in cluster_dict:
            # remove c0 key
            cluster_dict[c1] += cluster_dict[c0]
            cluster_dict.pop(c0)
        # add the speaker of the new cluster
        cluster_list.append(c1)
        ldiar.rename('cluster', [c0], c1)
        i += 1

    return ldiar, cluster_dict, merge

# def hac_update_model(diarization, ivectors):


def hac_iv_it(diar, model_iv, threshold=0.0):
    model_iv_local = copy.deepcopy(model_iv)
    model_iv_local.diar = copy.deepcopy(diar)
    np.fill_diagonal(model_iv_local.scores.scoremat, -np.inf)

    nb = model_iv_local.scores.modelset.shape[0]

    i, j, v = argmax(model_iv_local.scores.scoremat, nb)
    nb_merge = 0
    while v > threshold and nb > 1:
        nb_merge += 1
        logging.info('merge: %d c1: %s (%d) c2: %s (%d) dist: %f, size: %d',
                     nb_merge, model_iv_local.scores.modelset[i], i,
                     model_iv_local.scores.modelset[j], j,
                     v, model_iv_local.scores.modelset.shape[0])
        name_i = model_iv_local.scores.modelset[i]
        name_j = model_iv_local.scores.modelset[j]

        model_iv_local.update(i, j)
        model_iv_local.diar.rename('cluster', [name_j], name_i)
        np.fill_diagonal(model_iv_local.scores.scoremat, -np.inf)

        nb = model_iv_local.scores.modelset.shape[0]
        i, j, v = argmax(model_iv_local.scores.scoremat, nb)


    return model_iv_local.diar

