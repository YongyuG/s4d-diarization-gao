import copy
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import logging


class DER:
    """ Class to compute a s4d error rate.
    Error is computed at a rate of 10ms.

    :attr uem_set: a set of index to evaluate
    :attr collar: is the no-score zone around reference speaker segment
    boundaries.  (Speaker Diarization output is not evaluated within +/- collar
    seconds of a reference speaker segment boundary.)
    :attr uem_set_collar: a set of index on with the collar is applied
    :attr length: the last index in the reference
    :attr ref_cluster_list: the list of reference cluster_list
    :attr ref_mat: the numpy.ndarry of reference indexes, each row correspond
    to a reference cluster (same order as ref_cluster_list)
    :attr hyp_cluster_list: the list of hypothesis cluster_list
    :attr hyp_mat: the numpy.ndarry of reference indexes, each row correspond
    to a hypothesis cluster (same order as ref_cluster_list)
    :attr conf: the confusion matrix
    :attr assigned: a numpy.ndarray given the association between the references and hypotheses
    :attr not_assigned: a list of hypothesis cluster_list not associated to a reference cluster

    """
    def __init__(self, hyp_diarization, ref_diarization, uem_diarization = None, collar=0, no_overlap=False):
        """

        :param hyp_diarization: a hypothesis Diar object
        :param ref_diarization: a reference Diar object
        :param uem_diarization: a uem Diar object
        :param collar: is the no-score zone around reference speaker segments
        :param no_overlap: to limit scoring to those time regions in which only
        a single speaker is speaking

        """
        uem = uem_diarization
        self.collar = collar
        self.match = None
        if uem_diarization is not None:
            uem = copy.deepcopy(uem_diarization)
            if len(uem.unique('cluster')) > 1:
                raise Exception('UEM contains more than one cluster')
            if len(uem.unique('show')) > 1:
                raise Exception('UEM contains more than one show')
            self.uem_set = set(uem.features())

        ref = copy.deepcopy(ref_diarization)
        ref.pack()
        lst = ref.unique('show')
        if len(lst) > 1:
            raise Exception('REF contains more than one show')
        else:
            self.show = lst[0]
        if uem_diarization is None:
            print('uem from ref')
            self.uem_set = set([i for i in range(min(ref.unique('start')), max(ref.unique('stop')))])

        if collar > 0:
            rem = list()
            for row in ref:
                rem += [i for i in range(row['start']-collar, row['start']+collar)]
                rem += [i for i in range(row['stop']-collar, row['stop']+collar)]
            self.uem_set_collar = self.uem_set - set(rem)
        else:
            self.uem_set_collar = self.uem_set

        self.length = max(self.uem_set_collar)+1
        self.ref_cluster_list = ref.unique('cluster')
        ref_idx = ref.features_by_cluster()
        self.ref_mat = self._to_bool(ref_idx, self.ref_cluster_list, self.uem_set_collar)
        if no_overlap:
            print('remove overlap')
            sum = np.sum(self.ref_mat, axis=0)
            idx = np.squeeze(np.asarray(np.argwhere(sum == 1))).tolist()
            self.uem_set_collar = self.uem_set & set(idx)
            self.length = max(self.uem_set_collar)+1
            self.ref_mat = self._to_bool(ref_idx, self.ref_cluster_list, self.uem_set_collar)

        hyp = copy.deepcopy(hyp_diarization)
        hyp.pack()
        self.hyp_cluster_list = hyp.unique('cluster')
        hyp_idx = hyp.features_by_cluster()
        self.hyp_mat = self._to_bool(hyp_idx, self.hyp_cluster_list, self.uem_set_collar)

        self.conf = None
        self.assigned = None
        self.not_assigned = None

    def _to_bool(self, map, cluster_list, uem_set):
        """Transform the dict of segments indexed by cluster into a matrix
        of booleans.

        :param map: dict of segments indexed by speaker
        :param cluster_list: list of cluster_list
        :param uem_set: the segments is filtered according to the uem


        :return: a numpy.ndarray object of shape (#cluster, #features)
        """
        mat = np.zeros((len(cluster_list), self.length))

        for i, cluster in enumerate(cluster_list):
            mat[i, list(set(map[cluster]) & uem_set)] = True
        return mat

    def confusion(self):
        """Compute the confusion matrix between reference and hypothesis

        :return: a numpy.ndarray object of shape (#reference, #hypothesis)
        """
        #nh = len(self.hyp_cluster_list)
        #nr = len(self.ref_cluster_list)
        #self.conf = np.full((nr, nh), 0.0)
        #for i in range(nr):
        #    for j in range(nh):
        #        self.conf[i, j] = np.sum(np.logical_and(self.ref_mat[i], self.hyp_mat[j]))

        self.conf = np.dot(self.ref_mat, self.hyp_mat.transpose())

        return self.conf

    def confusion2str(self, all_value=False):
        """Generate a string version of the confusion matrix

        :param all_value: if false, only the confision score upper to 0 is
        generated

        :return: a str
        """
        mr = mh = 0
        for ch in self.ref_cluster_list+['reference']:
            if mr < len(ch):
                mr = len(ch)
        for ch in self.hyp_cluster_list+['hypothesis']:
            if mh < len(ch):
                mh = len(ch)

        f = '{:'+str(mr)+'} {:'+str(mh)+'} {:>9}\n'
        line = f.format('reference', 'hypothesis', 'conf')
        f = '{:'+str(mr)+'} {:'+str(mh)+'} {:>8.2f}s\n'
        nh = len(self.hyp_cluster_list)
        nr = len(self.ref_cluster_list)

        for i in range(nr):
            for j in range(nh):
                if self.conf[i, j] > 0 or all_value == True:
                    r = self.ref_cluster_list[i]
                    h = self.hyp_cluster_list[j]
                    s = self.conf[i, j]/100
                    line += f.format(r, h, s)
        return line

    def assignment(self):
        """Compute the reference to hypothese association using the hungarian
        algorithm

        :return: a dict object with key equal to (reference cluster, hypothesis cluster)
        and value equal to the confusion score
        :return: a list object containing the hypothesis cluster_list that are not assigned
        """
        self.assigned = linear_assignment(-1.0*self.conf)
        name_assigned = dict()
        self.match = 0;
        self.not_assigned = copy.deepcopy(self.hyp_cluster_list)

        for i in range(self.assigned.shape[0]):
            r = self.ref_cluster_list[self.assigned[i, 0]]
            h = self.hyp_cluster_list[self.assigned[i, 1]]
            s = self.conf[self.assigned[i, 0], self.assigned[i, 1]]
            if s > 0:
                self.not_assigned.remove(h)
                name_assigned[(r, h)] = s
                self.match += s
            else:
                self.assigned[i, 0] = -1
                self.assigned[i, 1] = -1

        return name_assigned, self.not_assigned

    def set_assignment(self, name_assigned):
        """Set assignment from an external source (from a collection of cluster_list)

        :param: name_assigned is a dict object with key equal to
        (reference cluster, hypothesis cluster) and value equal to the confusion
        score
        """
        i = 0;
        self.assigned = np.ones((len(self.ref_cluster_list), 2)) * -1

        for key in name_assigned:
            # print(key)
            if key[0] in self.ref_cluster_list and key[1] in self.hyp_cluster_list:
                r = self.ref_cluster_list.index(key[0])
                h = self.hyp_cluster_list.index(key[1])
                self.assigned[i, 0] = r
                self.assigned[i, 1] = h
                i += 1

    def assignment2str(self):
        """Generate a string version of the assignment matrix attribut

        :return: a str
        """
        if self.assigned is None:
            return "Assignment is not set"
        mr = mh = 0

        for ch in self.ref_cluster_list+['reference']:
            if mr < len(ch):
                mr = len(ch)
        for ch in self.hyp_cluster_list+['hypothesis']:
            if mh < len(ch):
                mh = len(ch)
        f = '{:'+str(mr)+'} {:'+str(mh)+'} {:>9}\n'
        line = f.format('reference', 'hypothesis', 'matching')
        f = '{:'+str(mr)+'} {:'+str(mh)+'} {:>8.2f}s {:d} {:d}\n'

        for i in range(self.assigned.shape[0]):
            if self.assigned[i, 0] >= 0:
                r = self.ref_cluster_list[self.assigned[i, 0]]
                h = self.hyp_cluster_list[self.assigned[i, 1]]
                s = self.conf[self.assigned[i, 0], self.assigned[i, 1]]/100
                line += f.format(r, h, s, self.assigned[i, 0], self.assigned[i, 1])

        r = 'None'
        s = 0.0
        for h in self.not_assigned:
            line += f.format(r, h, s, -1, -1)

        return line

    def _correct(self):
        """compute the correct decision

        :return: a numpy.ndarray
        """
        c = np.zeros(self.length)
        for i in range(self.assigned.shape[0]):
            if self.assigned[i, 0] >= 0 and self.assigned[i, 1] >= 0:
                diff = np.logical_and(self.ref_mat[self.assigned[i, 0]], self.hyp_mat[self.assigned[i, 1]])
                c += diff
        return c

    def error(self):
        """compute the s4d error

        :return: a DER_result object
        """
        h_sum = np.sum(self.hyp_mat, axis=0)
        r_sum = np.sum(self.ref_mat, axis=0)
        c = self._correct()

        res = DER_result(self.show)
        res.uem_time = len(self.uem_set)
        res.uem_with_collar_time = len(self.uem_set_collar)
        res.uem_collar_time = res.uem_time - res.uem_with_collar_time

        z = np.zeros(r_sum.shape[0])

        mrh = np.sum(np.maximum(r_sum - h_sum, z))
        mhr = np.sum(np.maximum(h_sum - r_sum, z))

        res.speech_time = np.count_nonzero(r_sum)
        res.sns_miss_time = np.count_nonzero(np.logical_and(r_sum, np.logical_not(h_sum)))
        res.sns_fa_time = np.count_nonzero(np.logical_and(h_sum, np.logical_not(r_sum)))
        res.spk_time = r_sum.sum()
        res.spk_miss_time = mrh
        res.spk_fa_time = mhr
        res.spk_conf_time = np.sum(np.minimum(h_sum, r_sum) - c)

        # for i in range(c.shape[0]):
        #     ref = r_sum[i]
        #     hyp = h_sum[i]
        #     mrh = max(ref - hyp, 0)
        #     mhr = max(hyp - ref, 0)
        #     res.speech_time += bool(ref)
        #     res.sns_miss_time += bool(ref and not hyp)
        #     res.sns_fa_time += bool(hyp and not ref)
        #     if ref > 1:
        #         res.overlap_time += ref
        #         res.overlap_miss_time += mrh
        #         res.overlap_fa_time += mhr
        #     res.spk_time += ref
        #     res.spk_miss_time += mrh
        #     res.spk_fa_time += mhr
        #     res.spk_conf_time += min(hyp, ref) - c[i]
        res.compute_rate()
        return res

class DER_coll(DER):
    def __init__(self, ref_cluster_list, hyp_cluster_list):
        self.ref_cluster_list = ref_cluster_list
        self.hyp_cluster_list = hyp_cluster_list
        hyp_nb_clusters = len(hyp_cluster_list)
        ref_nb_clusters = len(ref_cluster_list)
        self.conf = np.full((ref_nb_clusters, hyp_nb_clusters), 0.0)

    def merge_confusion(self, der):
        for i in range(len(der.ref_cluster_list)):
            for j in range(len(der.hyp_cluster_list)):
                v = der.conf[i, j]
                if v > 0:
                    ig = self.ref_cluster_list.index(der.ref_cluster_list[i])
                    jg = self.hyp_cluster_list.index(der.hyp_cluster_list[j])
                    self.conf[ig, jg] += v

class DER_result:
    """
    Class to store the s4d error results
    """
    def __init__(self, show):
        """

        :param show: a string
        """
        self.show = show

        self.uem_time = 0
        self.uem_with_collar_time = 0
        self.uem_collar_time = 0
        self.spk_time = 0
        self.speech_time = 0
        self.sns_time = 0
        self.overlap_time = 0
        self.overlap_miss_time = 0
        self.overlap_fa_time = 0
        self.spk_conf_time = 0
        self.sns_fa_time = 0
        self.sns_miss_time = 0
        self.overlap_miss = 0
        self.overlap_fa = 0
        self.spk_miss_time = 0
        self.spk_fa_time = 0
        self.spk_miss = 0
        self.spk_fa = 0
        self.spk_conf = 0
        self.speech_fa = 0
        self.speech_miss = 0
        self.len_line = 0

    def accumulate(self, result):
        """
        Accumulate the scores
        :param result: a DER_result object to accumulate
        :return:
        """
        self.uem_time += result.uem_time
        self.uem_with_collar_time += result.uem_with_collar_time
        self.uem_collar_time += result.uem_collar_time
        self.spk_time += result.spk_time
        self.speech_time += result.speech_time
        self.sns_time += result.sns_time
        self.overlap_time += result.overlap_time
        self.overlap_miss_time += result.overlap_miss_time
        self.overlap_fa_time += result.overlap_fa_time
        self.spk_conf_time += result.spk_conf_time
        self.sns_fa_time += result.sns_fa_time
        self.sns_miss_time += result.sns_miss_time
        self.overlap_miss += result.overlap_miss
        self.overlap_fa += result.overlap_fa
        self.spk_miss_time += result.spk_miss_time
        self.spk_fa_time += result.spk_fa_time
        self.spk_miss += result.spk_miss
        self.spk_fa += result.spk_fa
        self.spk_conf += result.spk_conf
        self.speech_fa += result.speech_fa
        self.speech_miss += result.speech_miss

    def compute_rate(self):
        """
        Compute the various rates
        """
        self.sns_fa = self.sns_fa_time / self.uem_time * 100
        self.sns_miss = self.sns_miss_time / self.uem_time * 100

        if self.overlap_time > 0:
            self.overlap_fa = self.overlap_fa_time / self.overlap_time * 100
            self.overlap_miss = self.overlap_miss_time / self.overlap_time * 100
        else:
            self.overlap_fa = 0
            self.overlap_miss = 0

        self.spk_fa = self.spk_fa_time / self.spk_time * 100
        self.spk_miss = self.spk_miss_time / self.spk_time * 100
        self.spk_conf = self.spk_conf_time / self.spk_time * 100

    @classmethod
    def header(cls, speech=True, overlap=False, speaker=True):
        """
        Make the header
        :param speech: add speech/non-speech results
        :param overlap: add overlap result
        :param speaker: add speaker result (DER)
        :return: a str
        """
        line = ['show', 'type']
        if speech:
            line += ['fa', 'miss', 'sns']
        if overlap:
            line += ['fa', 'miss', 'overlap']
        if speaker:
            line += ['fa', 'miss', 'conf', 'speaker']

        return [line]

    def rate(self, name=None, speech=True, overlap=False, speaker=True):
        """
        Make the rate results
        :param speech: add speech/non-speech results
        :param overlap: add overlap result
        :param speaker: add speaker result (DER)
        :return: a str
        """
        if name is None:
            name = ''
        line = [name, 'rate']
        if speech:
            line += [self.sns_fa, self.sns_miss, self.sns_fa+self.sns_miss]
        if overlap:
            line += [self.overlap_fa, self.overlap_miss, self.overlap_fa+self.overlap_miss]
        if speaker:
            line += [self.spk_fa, self.spk_miss, self.spk_conf, self.get_der()]
        return [line]

    def time(self, name=None, speech=True, overlap=False, speaker=True):
        """
        Make the time results
        :param speech: add speech/non-speech results
        :param overlap: add overlap result
        :param speaker: add speaker result (DER)
        :return: a str
        """
        if name is None:
            name = ''
        line = [name, 'time']
        if speech:
            line += [self.sns_fa_time/100, self.sns_miss_time/100, self.uem_with_collar_time/100]
        if overlap:
            line += [self.overlap_fa_time/100, self.overlap_miss_time/100, self.overlap_time/100]
        if speaker:
            line += [self.spk_fa_time/100, self.spk_miss_time/100, self.spk_conf_time/100, self.spk_time/100]
        return [line]

    def get_table(self, name=None, time=True, rate=True, overlap=False):
        lst = []
        add_name = False
        if rate:
            lst += self.rate(name=name, overlap=overlap)
            add_name = True

        if time:
            if not add_name:
                lst += self.time(name=name, overlap=overlap)
            else:
                lst += self.time(name=None)

        #if add_name == False and overlap:
        #    lst += self.overlap(speaker=speaker)
        #    add_name = True
        #else:
            #lst += self.overlap(speaker=None)
        return lst

    def get_der(self):
        return self.spk_fa+self.spk_miss+self.spk_conf


def get_header():
    return DER_result.header()


def compute_der(hyp, ref, uem=None, collar=0, no_overlap=True):
    cder = DER(hyp, ref, uem, collar=collar, no_overlap=no_overlap)
    cder.confusion()
    cder.assignment()
    return cder.error()
