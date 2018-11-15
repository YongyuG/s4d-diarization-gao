import os
from ecyglpki import Problem, SimplexControls
from scipy.cluster import hierarchy as hac
import matplotlib.pyplot as plt
from scipy import stats
import logging
import time
import os
from s4d.clustering.hac_utils import *

class ILP_IV:
    def __init__(self, diar, scores, threshold=0.0):
        self.sep = '!!'
        self.dic = dict()
        self.diar = copy.deepcopy(diar)

        self.thr = threshold
        self.scores = copy.deepcopy(scores)

    def plot(self, link, cluster_list):
        #print(link)
        hac.dendrogram(link, color_threshold=self.thr, labels=cluster_list)
        plt.show()

    def perform(self, save_ilp=False):
        """
        Creates an LP problem and solves it using GLPK library 
        (with ecyglpki Cython interface)
        :param save_ilp: if True, saves the generated problem
        in CPLEX LP format (problem.ilp), and the solution of
        the problem (solution.ilp).
        :return: a Diar object with the new clusters and a 
        dictionary of clusters (old clusters as key, new ones
        as value in an array).
        """
        distances, t = scores2distance(self.scores, self.thr)
        cluster_list = sorted(self.scores.modelset.tolist())

        lp = Problem()
        lp.set_obj_dir('minimize')

        # columns
        cols = []
        for idx, cluster in enumerate(cluster_list):
            col = '{}{}{}'.format(cluster, self.sep, cluster)
            cols.append(col)
            lp.add_named_cols(col)
            lp.set_obj_coef(col, 1)
            lp.set_col_bnds(col, 0, None)

        # sum of dist > thr in the lower triangular part od the distance matrix
        mask = (np.tril(distances, -1) > t)
        threshold = np.tril(distances, -1).copy()
        threshold[mask] = 0
        s = np.sum(threshold) + 1
        l = len(cluster_list)
        for i in range(l):
            cluster_i = cluster_list[i]
            for j in range(i + 1, l):
                if distances[i, j] < t:
                    cluster_j = cluster_list[j]
                    v = distances[i, j] / s
                    col = '{}{}{}'.format(cluster_i, self.sep, cluster_j)
                    cols.append(col)
                    lp.add_named_cols(col)
                    lp.set_obj_coef(col, v)
                    lp.set_col_bnds(col, 0, None)

        # rows
        for i, cluster_i in enumerate(cluster_list):
            r_cols = {}
            row = 'S{}'.format(i)
            lp.add_named_rows(row)
            r_cols['{}{}{}'.format(cluster_i, self.sep, cluster_i)] = 1

            for j, cluster_j in enumerate(cluster_list):
                if i != j and distances[i, j] < t:
                    col = '{}{}{}'.format(cluster_i, self.sep, cluster_j)
                    if col not in cols:
                        cols.append(col)
                        lp.add_named_cols(col)
                        lp.set_col_bnds(col, 0, None)
                    r_cols[col] = 1

                    #boundaries <= 0
                    col = '{}{}{}'.format(cluster_i, self.sep, cluster_j)
                    idx = lp.add_rows(1)
                    lp.set_mat_row(idx, {'{}{}{}'.format(cluster_j, self.sep, cluster_j):-1, col:1})
                    lp.set_row_bnds(idx, None, 0)
                    
            lp.set_mat_row(row, r_cols)
            lp.set_row_bnds(row, 1, 1)

        if save_ilp:
            lp.write_lp('problem.ilp')

        # solving problem
        ctrl = SimplexControls()
        ctrl.presolve = True
        lp.simplex(ctrl)

        if save_ilp:
            lp.print_sol('solution.ilp')

        cluster_dict = dict()
        for i in range(lp.get_num_cols()):
            names = lp.get_col_name(i+1).split(self.sep)
            activity = lp.get_col_prim(i+1)
            if activity == 1 and names[0] != names[1]:
                if names[1] not in cluster_dict:
                    cluster_dict[names[1]] = []
                cluster_dict[names[1]].append(names[0])

        table = copy.deepcopy(self.diar)
        for idx in cluster_dict:
            table.rename('cluster', cluster_dict[idx], idx)
        return table, cluster_dict



    def _perform(self, filename='tmp.ilp', rm_tmp=True):
        """
        Same as perform(), using glpk solver directly
        instead of the Cython interface (slower).
        """
        table = copy.deepcopy(self.diar)
        logging.debug('ilp filename: %s', filename)
        f = open(filename, 'w')
        self._ilp_write(f)
        f.close()
        while not os.path.exists(filename):
            time.sleep(1)
        cmd = 'glpsol --lp {} -o {} &> {}'.format(filename, filename + '.out', filename + '.err')
        #print(cmd)
        if os.path.exists(filename + '.out'):
            os.remove(filename + '.out')
            while os.path.exists(filename + '.out'):
                time.sleep(1)
        os.system(cmd)
        time.sleep(1)
        while not os.path.exists(filename + '.out'):
            time.sleep(1)
        f = open(filename + '.out', 'r')
        cluster_dict = self._ilp_read(f)
        f.close()
        for idx in cluster_dict:
            table.rename('cluster', cluster_dict[idx], idx)
        if rm_tmp:
            os.remove(filename)
            os.remove(filename+ '.out')
            os.remove(filename+ '.err')
            while os.path.exists(filename) or os.path.exists(filename+ '.out') or os.path.exists(filename+ '.err'):
                time.sleep(1)
        return table, cluster_dict

    def _ilp_write(self, f):
        distances, t = scores2distance(self.scores, self.thr)

        cluster_list = self.scores.modelset.tolist()
        f.write("Minimize\n")
        f.write("problem : \n")
        for idx, cluster in enumerate(cluster_list):
            f.write(' + {}{}{}'.format(cluster, self.sep, cluster))
        # sum of dist > thr in the lower triangular part od the distance matrix

        mask = (np.tril(distances, -1)>t)
        threshold = np.tril(distances, -1).copy()
        threshold[mask] = 0
        s = np.sum(threshold) + 1
        #s = np.sum(stats.threshold(np.tril(distances, -1), threshmax=t, newval=0)) + 1
        logging.debug('ilp sum scores: '+str(s))
        l = len(cluster_list)
        for i in range(l):
            cluster_i = cluster_list[i]
            for j in range(i+1, l):
                if distances[i, j] < t:
                    cluster_j = cluster_list[j]
                    v = distances[i, j] / s
                    f.write(
                        ' +{} {}{}{} '.format(v, cluster_i, self.sep,
                                                cluster_j))
        f.write("\nSubject to\n")

        # x_i,i is a centre or not
        for i, cluster_i in enumerate(cluster_list):
            f.write('S{}: {}{}{}'.format(i, cluster_i, self.sep, cluster_i))
            for j, cluster_j in enumerate(cluster_list):
                if i != j and distances[i, j] < t:
                    f.write(" + {}{}{}".format(cluster_i, self.sep, cluster_j))
            f.write(' = 1\n')

        # center
        for i, cluster_i in enumerate(cluster_list):
            for j, cluster_j in enumerate(cluster_list):
                if i != j and distances[i, j] < t:
                    f.write("{}{}{} - {}{}{} <=0\n".format(cluster_i, self.sep, cluster_j,
                                                           cluster_j, self.sep, cluster_j))
        f.write('End')

    def _ilp_read(self, f):
        cluster_dict = dict()
        for line in f:
            line.strip()
            if line.find(self.sep) != -1:
                lst = line.split()
                if len(lst) < 3:
                    line += f.read()
                    lst = line.split()
                names = lst[1].split(self.sep)
                if lst[3] == '1' and names[0] != names[1]:
                    logging.debug('merge %s in %s', names[0], names[1])
                    if names[1] not in cluster_dict:
                        cluster_dict[names[1]] = []
                    cluster_dict[names[1]].append(names[0])
        return cluster_dict


def ilp_iv(diar, scores, threshold=0.0):
    ilp = ILP_IV(diar, scores, threshold)
    return ilp.perform()
