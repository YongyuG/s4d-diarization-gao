__author__ = 'meignier'

import copy
import logging
import numpy as np
from scipy.sparse import csgraph
from collections import namedtuple
from sidekit.bosaris.scores import Scores
from s4d.clustering.hac_utils import *


ConnectedComponentTuple = namedtuple('ConnectedComponent', ['type', 'diarization', 'scores' ])


StarGraphTuple = namedtuple('ConnectedComponent', ['type', 'diarization', 'scores',
                                                     'center', 'within_inertia'])


class ConnectedComponent:
    def __init__(self, diar, scores, threshold):
        """
        :param diarization: s4d annotation
        :param cluster_list: list of cluster_list, the ith cluster corresponds to the speaker
            of column and row i in distances_
        :param distances: matrice of distances between cluster_list (speakers)
        :param thr: the threshold will be apply to the distances_ to generate a graph
        """
        # output: the list of CC, each row contains the sub-diarization, the sub-matrix
        #  and the cluster list of the cc
        self.cc = list()
        # output: contain the star graphs (and isolated vertices)
        self.diar = copy.deepcopy(diar)
        self.scores = copy.deepcopy(scores)
        self.thr = threshold
        self.cc_nb = -1
        self.cc_list = None
        self.nb_sg = 0
        self.nb_sg0 = 0
        self.n = 0


    def _star_graph(self, lst, graph):
        """
        Verify if a star graph exists in the sub graph
        (http://en.wikipedia.org/wiki/Star_(graph_theory))
        :param lst: list of vertices
        :param graph: the graph
        :return the center and the within_inertia
        """
        within_inertia = np.inf
        center = -1
        for i in lst:
            s = np.sum(graph[i, lst])
            if within_inertia > s:
                within_inertia = s
                center = i
        return center, within_inertia

    def sub_graph(self, rename_cc=False):
        """
        find the 2 kind of sub-graphs:
         - isolated vertices and star graphs already clustered
         - other connected components put in self.cc list
        :return: generate self.cc and self.table_out
        """
        logging.debug('threshold the distance matrix')
        distances, t = scores2distance(self.scores, self.thr)

        mask = (distances>t)
        graph = distances.copy()
        graph[mask] = np.inf
        #graph = threshold(distances, threshmax=t, newval=np.inf)
        logging.debug('get connected components')
        cc_nb, cc_list = csgraph.connected_components(graph, directed=False)
        diar_out = copy.deepcopy(self.diar)
        # list of lists, each sub list contains the index of a connected component
        # print(cc_nb, cc_list)
        lst = []
        for i in range(cc_nb):
            lst.append(list())

        for j in range(len(cc_list)):
            lst[cc_list[j]].append(j)
        self.cc_list = list()
        # for each connected component
        for slst in lst:
            # start graph if the list contains only one index
            #print(slst)
            scores = Scores()
            scores.modelset = np.copy(self.scores.modelset[slst])
            scores.segset = np.copy(self.scores.segset[slst])
            scores.scoremask = np.copy(self.scores.scoremask[slst, :][:, slst])
            scores.scoremat = np.copy(self.scores.scoremat[slst, :][:, slst])

            diar_cc = self.diar.filter('cluster', 'in', scores.modelset.tolist())
            if len(slst) == 1: # isolated vertex
                cc = StarGraphTuple('star_graph_0', diar_cc, scores, self.scores.modelset[slst[0]], 0)
                self.nb_sg0 += 1
            else:
                center, within_inertia = self._star_graph(slst, graph)
                if center >= 0: # Star Graph if the list contains a center
                    l = diar_cc.unique('cluster')
                    #logging.info('nb_sg rename '+str(l)+' into '+self.scores.modelset[center])
                    diar_out.rename('cluster', l, self.scores.modelset[center])
                    diar_cc.rename('cluster', l, self.scores.modelset[center])
                    cc = StarGraphTuple('star_graph_k', diar_cc, scores, self.scores.modelset[center], within_inertia)
                    self.nb_sg += 1
                else: # connected component without star graph
                    if rename_cc:
                        diar_out.rename('cluster', diar_cc.unique('cluster'), self.scores.modelset[center])
                        #diar_cc.rename('cluster', diar_cc.unique('cluster'), self.scores.modelset[center])
                    cc = ConnectedComponentTuple('cc', diar_cc, scores)
                    self.n += 1

            self.cc_list.append(cc)
        logging.debug('-- stat CC '+str(self.thr)+' -- nb_sg0: '+str(self.nb_sg0)
                     +' nb_sg: '+str(self.nb_sg)+' cc: '+str(self.n)+ ' nb: '
                     +str(self.n + self.nb_sg0 + self.nb_sg)+'/'+str(cc_nb))
        return diar_out, self.cc_list, self.nb_sg0, self.nb_sg, self.n


def connexted_component(diar, scores, threshold):

    graphs = ConnectedComponent(diar, scores, threshold)
    return graphs.sub_graph()

