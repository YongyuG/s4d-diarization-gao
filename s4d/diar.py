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

"""
Diar is a class describing an audio/video segmentation file. A diarization
contains a list of segments. Where each row is segment composed of n values
identified by a attribut names.
The diarization file is the most important file in the toolkit. All programs
are driven by a diarization file and most of them generate a diarization
file (trainer generate gmm).

A diarization stores a list of ''segments'' composed of attributes.

A diarization could draw data from several shows. It is very useful in a batch
mode context (training of GMM, computing log likelihood ratio, cross-show
diarization, etc.).

Example
-------
>>> diarization[0] //get the first segment
['20041006_0700_0800_CLASSIQUE', 'Emmanuel_Cugny', 'speaker', 164, 1170]
>>> diarization[0]['show']
'20041006_0700_0800_CLASSIQUE'
>>> diarization[0]['cluster']
'Emmanuel_Cugny'
>>> diarization[0]['start']
164
>>> diarization[0]['stop']
1170

where:
  * attribut 0 named ''show'': ''20041006_0700_0800_CLASSIQUE'' = the show speaker
  * attribut 1 named ''cluster'' : ''Emmanuel_Cugny'' the speaker speaker
  * attribut 2 named ''type'' : ''speaker'' contains the cluster type (speaker or head)
  * attribut 3 named ''start'': ''164'' is the index of the first feature of the segment
  * attribut 4 named ''stop'': ''1170'' is the index of the last feature of the segment

How to
------
* Read a diarization:
    ::

        from s4d.diarization import Diar
        diarization = Diar.read_seg('foo.seg') //LIUM Spk Diarization format
        diarization = Diar.read_mdtm('foo.seg') //MDTM format
        diarization = Diar.read_rttm('foo.seg') //RTTM format
        diarization = Diar.read_uem('foo.seg') //UEM format

* Get a segment:
    ::

        seg = diarization[0]

* Get a attribut of a segment
    ::

        seg['cluster']

* Write a diarization:
    ::

        diarization = Diar.write_seg('foo.seg', diarization) //LIUM format

* Add or remove an attribut:
    ::

        diarization.add_attribut(speaker='gender', default=None) // add an attribut named 'gender, the default value is None
        diarization.del_attribut('gender') // remove the attribut

* Sort a diarization:
    ::

        diarization.sort()

* Create a new segment:
    ::

        diarization.append(show='foo', cluster='speaker', start=0, stop=100)

Add all the segments of diar2 into diar1:
    ::

        diar1.append_diar(diar2)

Modules
-------

"""

from sidekit.sidekit_wrappers import *
import re as regexp
import copy
import re
import logging
import os
from sidekit.bosaris.idmap import IdMap
import sys
from six import string_types
from s4d.utils import str2str_normalize

try:
    from sortedcontainers import SortedDict as dict
except ImportError:
    pass


class Diar():
    """
    The diarization class.

    :attr _attributes: a AttributeNames object storing the attribut definitions
    :attr cluster_types: a list object
    :attr segments: a list of Segment object
    """
    def __init__(self):
        self._attributes = AttributeNames()
        self._attributes.initialize({'show': 0, 'cluster': 1, 'cluster_type': 2,
                                  'start': 3, 'stop': 4},
                                    ['empty', 'empty', 'speaker', 0, 0])
        self.cluster_types = ['speaker', 'head']
        self.segments = list()

    def copy_structure(self):
        """
        Copy the internal structure of the diarization, ie the attribut names
        and the cluster types. The data is not copy.
        :return: a Diar object
        """
        tmp_diarization = Diar()
        tmp_diarization._attributes = copy.deepcopy(self._attributes)
        tmp_diarization.cluster_types = copy.deepcopy(self.cluster_types)
        return tmp_diarization

    def del_all(self, attribute, value):
        """
        Delete all segments satisfing the boolean expression [attribute = value]
        :param attribute: speaker of the attribute to delete
        :param value:
        :return:
        """
        lst = list()
        for segment in self.segments:
            if segment[attribute] != value:
                lst.append(segment)
        self.segments = lst

    def overlap(self, add_intersection=False):
        """
        remove overlap zone
        :return: a new diarization without overlap
        """

        def add(_show, _features_index, _cluster_list, _uem):
            diar_tmp = self.copy_structure()
            for lcluster in _cluster_list:
                lst = sorted(_uem.intersection(set(_features_index[lcluster])))
                if len(lst) > 0:
                    c = lst[0];
                    diar_tmp.append(show=_show, start=c, stop=c+1, cluster=lcluster)
                    l = 0
                    for j in range(1, len(lst)):
                        p = c
                        c = lst[j]
                        if c == p + 1:
                            l += 1
                        else:
                            diar_tmp[-1]['stop'] += l
                            diar_tmp.append(show=_show, start=c, stop=c+1, cluster=lcluster)
                            l = 0
                    if l > 0:
                        diar_tmp[-1]['stop'] += l
            return diar_tmp

        diarization_out = self.copy_structure()
        #shows = self.unique('show')
        shows = self.make_index(['show'])
        for show in shows:
            logging.info('rm overlap show: '+show)
            diar_show = shows[show]
            length = diar_show.last_feature_index()
            cluster_list = diar_show.unique('cluster')
            mat = numpy.zeros(length)
            features_index = diar_show.features_by_cluster()
            for i, cluster in enumerate(cluster_list):
                mat[features_index[cluster]] += 1
            uem = set(numpy.where(mat == 1)[0].tolist())
            diarization_out.segments += add(show, features_index, cluster_list, uem)
            if add_intersection:
                uem = set(numpy.where(mat != 1)[0].tolist())
                diarization_out.segments += add(show, features_index, cluster_list, uem)

        return diarization_out

    def filter(self, attribute, operator, value):
        """
        build a new diarization whose segments satisfy the boolean expression
        [attribute operator value]
        :param attribute: a attribute speaker (str)
        :param operator: a comperator opertor (> < >= <= in == !=)
        :param value: the value (int, float, str, list...)
        :return: a Diar object
        """
        tmp_diarization = self.copy_structure()
        tmp_diarization.segments = list()
        if attribute == "length" or attribute == "duration":
            str = "seg.duration() {:s} {}".format(operator, value)
        elif isinstance(value, string_types):
            str = "seg['{:s}'] {:s} '{:s}'".format(attribute, operator, value)
        else:
            str = "seg['{:s}'] {:s} {} ".format(attribute, operator, value)
        # print(ch)
        logging.debug(str)
        for seg in self.segments:
            # print(ch, seg.length())
            if eval(str):
                tmp_diarization.segments.append(copy.deepcopy(seg))
        return tmp_diarization

    def rename(self, attribute, old_values, new_value):
        """
        Rename all values in list old_values into the new value new_value
        :param attribute: speaker of the attribute
        :param old_values:  list of old values
        :param new_value: new value

        """
        for segment in self.segments:
            if segment[attribute] in old_values or len(old_values) == 0 :
                segment[attribute] = new_value

    def _iofi(self, index, attributes, segment):
        """
        recursive fonction to add a segment into the n level keys dictionary

        :param index: dict object of level n
        :param attributes: list of attribut attributes
        :param segment: a segment
        :return: a dictornary of level n that contains sub diarization. Segments
        are not copy.
        """
        # removes and gets the last attribut speaker
        attribut = attributes.pop()
        # takes the values of this attribut speaker
        value = segment[attribut]
        # if there is no more attribut attributes
        if len(attributes) <= 0:
            if value in index:
                # add the segment to the list
                index[value].append_seg(segment)
            else:
                # create a list and add the segment
                ldiar = self.copy_structure()
                ldiar.append_seg(segment)
                index[value] = ldiar
            return index
        else:
            # recursion to the level n-1 until attributes is empty
            self._iofi(index[value], attributes, segment)

    def make_index(self, attributes):
        """
        Build a n level key dictionary (dictionary of dictionaries of
        dictionaries...) based on Index.
        Index is an implementation of perl's autovivification feature.
        The values contains a list of row.

        example :

            d = make_index(['show', 'gender', 'cluster'])

            print(d['show1']['M']['speaker'])

        :param attributes: a list of attribut _attributes corresponding to the key indexes
        :return: a dictionary of sub diarization. Segments are not copy.
        """
        index = Index()
        for segment in self.segments:
            self._iofi(index, attributes[::-1], segment)
        return index

    def unique(self, attibute):
        """
        :param attibute: the attibute of the attribut
        :return: a list object of unique value of the attribut
        """
        dic = dict()
        lst = list()
        for seg in self.segments:
            dic[seg[attibute]] = 0
        for value in dic.keys():
            lst.append(value)
        return lst

    def sort(self, attributes=['show', 'start'], reverse=False):
        """
        Sort the segments
        :param attributes: a list of attribut names
        :param reverse: if true, make a reverse sort

        """
        attributes.reverse()
        for attribute in attributes:
            if attribute not in self._attributes:
                raise Exception("This attribut don't exits : " + attribute)
            self.segments = sorted(self.segments, key=lambda x: x[self._attributes[attribute]],
                                   reverse=reverse)

    def clear(self):
        """
        remove all the segments
        :return:
        """
        self.segments = list()

    def add_attribut(self, new_attribut, default=''):
        """
        Add a attribut
        :param new_attribut: the speaker of the new attribut
        :param default: the default value of the attribut

        """
        self._attributes.add(new_attribut, default)
        for seg in self.segments:
            seg.append(default)

    def del_attribut(self, attribut):
        """
        Delete a attribut
        :param attribut: the speaker of the attribut to detele

        """
        if attribut not in self._attributes:
            raise Exception("This attribut don't exits : " + attribut)
        else:
            i = self._attributes[attribut]
            for seg in self.segments:
                del seg[i]
            self._attributes.delete(attribut)

    def _new_row(self, **kwargs):
        """
        Create a new segment initialized with kwargs
        :param kwargs: the values
        :return:
        """
        seg = Segment(self._attributes.defaults, self._attributes)
        for key, value in kwargs.items():
            seg[self._attributes[key]] = value
        return seg

    def append(self, **kwargs):
        """
        Transforme a list of values into a segment and append the segmnt into
        the existing segment list.
        :param kwargs: the values
        :return:
        """
        self.segments.append(self._new_row(**kwargs))

    def append_seg(self, segment):
        """
        Append a Segment object into the existing segment list.

        :param segment: a Segment object

        """
        self.segments.append(segment)

    def append_list(self, segment_lst):
        """
        Append a list of segments into the existing segment segment_lst.
        :param segment_lst: a list of segments

        """
        self.segments += segment_lst

    def append_diar(self, out_diarization):
        """
        Append a diarization.
        :param out_diarization: a diarization object

        """
        lMatchAttr=list()
        for i in self._attributes.names:
            if i in out_diarization._attributes.names:
                lMatchAttr.append(i)
        assert len(lMatchAttr)!=0,"No attribute matches"
        
        lSegments=list()
        for i in out_diarization:
            seg=Segment(self._attributes.defaults,self._attributes)
            for y in lMatchAttr:
                seg._set_attr(y,i[y])
            lSegments.append(seg)
        
        self.segments += lSegments

    def insert(self, i, **kwargs):
        """
        Insert values into the list at offset index
        :param i: This is the Index where the object obj need to be inserted.
        :param kwargs: the values

        """
        self.segments.insert(i, self._new_row(**kwargs))

    def __iter__(self):
        """
        This method is called when an iterator is required for a container.
        :return: an iterator
        """
        return self.segments.__iter__()

    def __reversed__(self):
        """
        Called (if present) by the reversed() built-in to implement reverse iteration.
        :return: a Diar object
        """
        return self.segments.__reversed__()

    def __delitem__(self, index):
        """
        Called to implement deletion of self[index]
        :param index: a int

        """
        del self.segments[index]

    def __getitem__(self, index):
        """
        Called to implement evaluation of self[index]
        :param index: a int
        :return: a Segment object
        """
        return self.segments[index]

    def __setitem__(self, index, value):
        """
        Called to implement evaluation of self[index] = value

        :param index: a int
        :param value:  a Segment object

        """
        self.segments[index] = value

    def __len__(self):
        """
        :return: the number of segments
        """
        return len(self.segments)

    def __eq__(self, diarization): # real signature unknown
        idx_self = self.make_index(['show', 'cluster', 'start'])
        idx = diarization.make_index(['show', 'cluster', 'start'])

        for show in idx_self:
            for cluster in idx_self[show]:
                for start in idx_self[show][cluster]:
                    if show not in idx:
                        return False
                    elif cluster not in idx[show]:
                        return False
                    elif start not in idx[show][cluster]:
                        return False
                    else :
                        l1 = idx[show][cluster][start].segments
                        l2 = idx_self[show][cluster][start].segments
                        return all(x in l2 for x in l1)
        return True

    def __ne__(self, diarization): # real signature unknown
        return not self.__eq__(diarization)

    def __repr__(self):
        """
        :return: a string version of the diarization
        """
        string = '  attribut definition  : ['
        index = 0
        lst = self._attributes.sorted()
        #print(lst)
        for attribute in lst:
            string += "'" + attribute[0] + "', "
        string = regexp.sub(', $', '', string) + ']\n'
        for segment in self.segments:
            line = ''
            for attribute in segment:
                line += attribute.__repr__() + ', '
            string += '  row ' + str(index) + ': [' + regexp.sub(', $', '',
                                                           line) + ']\n'
            index += 1
        return '[\n' + string + ']'

    def __add__(self, diarization):
        diarization_copy = copy.deepcopy(self)
        diarization_copy.segments += diarization.segments
        return diarization_copy

    def __iadd__(self, diarization):
        self.segments += diarization.segments
        return self

    def id_map(self, id_attribut='cluster', show_attribut='show',
               prefix_id_attrubut=None, suffix_show_attribut=None):
        """
        Generate a IdMap object for the StatServer
        :param id_attribut: speaker id_attribut attribut
        :param show_attribut: show_attribut attribut
        :param prefix_id_attrubut: prefix string of id_attribut
        :param suffix_show_attribut: suffix string of id_attribut

        :param out_diarization: a diarization object
        :return: a IdMap object
        """
        id_map = IdMap()
        id_map.leftids = numpy.empty(len(self.segments), dtype="|O")
        id_map.rightids = numpy.empty(len(self.segments), dtype="|O")
        id_map.start = numpy.empty(len(self.segments), dtype="|O")
        id_map.stop = numpy.empty(len(self.segments), dtype="|O")

        i = 0
        for segment in self.segments:
            if prefix_id_attrubut is not None:
                id_map.leftids[i] = segment[prefix_id_attrubut] + '/' + segment[id_attribut]
            else:
                id_map.leftids[i] = segment[id_attribut]
            if suffix_show_attribut is not None:
                id_map.rightids[i] = segment[show_attribut] + '/' + segment[suffix_show_attribut]
            else:
                id_map.rightids[i] = segment[show_attribut]
            id_map.start[i] = segment['start']
            id_map.stop[i] = segment['stop']
            i += 1

        return id_map

    def features_by_cluster(self, show=None, maximum_length=None):
        """
        Generate the indexes of a show
        :param show: the speaker of the show
        :param maximum_length: maximum length of the show
        :return: a dict object (keys are the cluster_list)
        """
        if show == None:
            l = self.unique('show')
            if len(l) > 1:
                raise Exception('diarization address sevreal shows, set show parameter')
            else:
                show = l[0]
        dic = dict()
        for segment in self.segments:
            if show == segment['show']:
                cluster = segment['cluster']
                start = segment['start']
                stop = segment['stop']
                if maximum_length is not None:
                    start = min(segment['start'], maximum_length)
                    stop = min(segment['stop'], maximum_length)

                if cluster not in dic:
                    dic[cluster] = []
                dic[cluster] += [i for i in range(start, stop)]
        return dic

    def features(self, show=None, maximum_length=None):
        """
        Generate the index features of a show
        :param show: a string corresponding to the speaker of the show
        :param maximum_length: maximum length of the show
        :return: a list object of indexes
        """
        if show is None:
            lst = self.unique('show')
            if len(lst) > 1:
                raise Exception('diarization address sevreal shows, set show parameter')
            else:
                show = lst[0]
        lst = list()
        for segment in self.segments:
            if show == segment['show']:
                start = segment['start']
                stop = segment['stop']
                if maximum_length is not None:
                    start = min(segment['start'], maximum_length)
                    stop = min(segment['stop'], maximum_length)
                lst += [i for i in range(start, stop)]
        return lst
    def to_list(self, show=None, uem_start=None, uem_stop=None):
        if show is None:
            lst = self.unique('show')
            if len(lst) > 1:
                raise Exception('diarization address sevreal shows, set show parameter')
            else:
                show = lst[0]
        if uem_start is None:
            uem_start = 0
        if uem_stop is None:
            uem_stop = self.last_feature_index()
        lst = [''] * uem_stop
        for segment in self.segments:
            if show == segment['show']:
                cluster = segment['cluster']
                start = max(segment['start'], uem_start)
                stop = min(segment['stop'], uem_stop)
                for i in range(start, stop):
                    if lst[i] == '':
                        lst[i] = cluster
                    else:
                        lst[i] += ' '+cluster
        return lst

    def pack(self, epsilon=0, coveringOverlap=False):
        """
        merge segments with a gap less than epsilon
        :param epsilon: a int value
        :param coveringOverlap: a boolean value
        """
        
        #index = self.make_index(['show'])
        #lst = list()
        #for show in index:
        #    diar = index[show]
        #    diar.sort(['start'])
        #    i = 0
        #    while i < len(diar.segments) - 1:
        #        if diar.segments[i]['cluster'] == diar.segments[i + 1]['cluster']:
        #            l = Segment.gap(diar.segments[i], diar.segments[i + 1]).duration()
        #            if l <= epsilon:
        #                diar.segments[i]['stop'] = max(diar.segments[i]['stop'],
        #                                               diar.segments[i + 1]['stop'])
        #                del diar.segments[i + 1]
        #            else:
        #                i += 1
        #        else:
        #            i += 1
        #    lst += diar.segments
        #self.segments = lst
        if coveringOverlap:
            index = self.make_index(['show', 'cluster'])
            lst = list()
            for show in index:
                for cluster in index[show]:
                    index[show][cluster].sort(['start'])
                    diar = index[show][cluster]
                    i = 0
                    while i < len(diar.segments) - 1:
                        l = Segment.gap(diar.segments[i], diar.segments[i + 1]).duration()
                        if l <= epsilon:
                            diar.segments[i]['stop'] = max(diar.segments[i]['stop'],
                                                           diar.segments[i + 1]['stop'])
                            del diar.segments[i + 1]
                        else:
                            i += 1
                    lst += diar.segments
            self.segments = lst
        else:
            self.sort(['show', 'start'])
            i = 0
            while i < len(self.segments) - 1:
                if self.segments[i]['show'] == self.segments[i + 1]['show'] and \
                        self.segments[i]['cluster'] == self.segments[i + 1]['cluster'] and \
                        Segment.gap(self.segments[i], self.segments[i + 1]).duration() <= epsilon:
                    self.segments[i]['stop'] = self.segments[i + 1]['stop']
                    del self.segments[i + 1]
                else:
                    i += 1

    def pad(self, epsilon=0):
        """
        Add epsilon frames to the start and stop of each segment
        :param epsilon: the int value to remove
        :return:
        """
        self.sort(['start'])
        i = 0
        if len(self.segments) > 1:
            self.segments[i]['stop'] = max(self.segments[i]['start'], min(max(self.segments[i + 1]['start'] - (epsilon // 2), 0), self.segments[i]['stop'] + epsilon))
        i += 1
        while i < len(self.segments)-1:
            self.segments[i]['start'] = max(self.segments[i - 1]['stop'], self.segments[i]['start'] - epsilon, 0)
            self.segments[i]['stop'] = max(self.segments[i]['start'],min(max(self.segments[i + 1]['start'] - (epsilon // 2), 0), self.segments[i]['stop'] + epsilon))
            i += 1

    def collar(self, epsilon=0, warning=False):
        """
        Apply a collar on each segment. A collar is the no-score zone around
        reference speaker segment boundaries.  (Speaker Diarization output is
        not evaluated within +/- collar seconds of a reference speaker segment
        boundary.)
        :param epsilon: the int value to add
        """
        self.sort(['start'])
        rm = False
        for segment in self.segments:
            segment['stop'] -= epsilon
            segment['start'] += epsilon
            if segment['start'] < 0:
                segment['start'] = 0
            if segment['start'] > segment['stop']:
                segment['start'] = segment['stop']
                rm = True
                if warning:
                    logging.warning('no more segment: '+str(segment['start']-epsilon))
        if rm:
            self.segments = [seg for seg in self.segments if seg.duration() > 0]

    def duration(self):
        """
        :return: the sum of the segment duration
        """
        l = 0
        for segment in self.segments:
            l += segment.duration()
        return l

    def last_feature_index(self):
        last = 0
        for segment in self.segments:
            if segment['stop'] > last:
                last = segment['stop']
        return last

    def first_feature_index(self):
        if len(self.segments) <= 0:
            return 0
        first = self.segments[0]['start']
        for segment in self.segments:
            if segment['start'] < first:
                first = segment['start']
        return first
    
    @classmethod
    def read_seg(cls, filename, normalize_cluster=False, encoding="utf8"):
        """
        Read a segmentation file
        :param filename: the str input filename
        :param normalize_cluster: normalize the cluster speaker by removing upper
        case and accents
        :return: a diarization object
        """
        fic = open(filename, 'r', encoding=encoding)
        diarization = Diar()
        if not diarization._attributes.exist('gender'):
            diarization.add_attribut(new_attribut='gender', default='U')
        if not diarization._attributes.exist('env'):
            diarization.add_attribut(new_attribut='env', default='U')
        if not diarization._attributes.exist('channel'):
            diarization.add_attribut(new_attribut='channel', default='U')
        try:
            for line in fic:
                line = re.sub('\s+',' ',line)
                line = line.strip()
                # logging.debug(line)
                if line.startswith('#') or line.startswith(';;'):
                    continue
                # split line into fields
                show, tmp, start, length, gender, channel, environment, name = line.split()
                if normalize_cluster:
                    name = str2str_normalize(name)
                # print(show, tmp, start, length, gender, channel, env, speaker)
                diarization.append(show=show, cluster=name, start=int(start),
                             stop=int(length) + int(start), env=environment,
                             channel=channel,
                             gender=gender)
        except Exception as e:
            logging.error(sys.exc_info()[0])
            # logging.error(line)
        fic.close()
        return diarization

    @classmethod
    def read_ctm(cls, filename, normalize_cluster=False, encoding="utf8"):
        """
        Read a segmentation file
        :param filename: the str input filename
        :param normalize_cluster: normalize the cluster by removing upper case
        and accents
        :return: a diarization object
        """
        fic = open(filename, 'r', encoding=encoding)
        diarization = Diar()
        try:
            for line in fic:
                line = re.sub('\s+',' ',line)
                line = line.strip()
                # logging.debug(line)
                if line.startswith('#') or line.startswith(';;'):
                    continue
                # split line into fields
                show, tmp, start, length, word = line.split()
                if normalize_cluster:
                    word = str2str_normalize(word)
                # print(show, tmp, start, length, gender, channel, env, speaker)
                diarization.append(show=show, cluster=word, start=int(start),
                             stop=int(length) + int(start))
        except Exception as e:
            logging.error(sys.exc_info()[0])
            # logging.error(line)
        fic.close()
        return diarization

    @classmethod
    def read_stm(cls,filename, normalize_cluster=False, encoding="ISO-8859-1"):
        """
        Read a segmentation file
        :param filename: the str input filename
        :param normalize_cluster: normalize the cluster by removing upper case
        and accents
        :return: a diarization object
        """
        fic = open(filename, 'r', encoding=encoding)
        diarization = Diar()
        if not diarization._attributes.exist('gender'):
            diarization.add_attribut(new_attribut='gender', default='U')
        try:
            for line in fic:
                line = re.sub('\s+',' ',line)
                line = line.strip()
                # logging.debug(line)
                if line.startswith('#') or line.startswith(';;'):
                    continue
                # split line into fields
                split = line.split()
                show = split[0]
                loc = split[2]
                if normalize_cluster:
                    loc = str2str_normalize(loc)
                start = int(float(split[3])*100)
                stop = int(float(split[4])*100)
                addon = split[5].replace(">", "").replace("<", "").replace(","," ")
                lineBis = re.sub('\s+',' ',addon)
                lineBis = lineBis.strip()
                gender = lineBis.split()[2]
                if normalize_cluster:
                    word = str2str_normalize(word)
                # print(show, tmp, start, length, gender, channel, env, speaker)
                if gender == "female":
                    diarization.append(show=show, cluster=loc, start=start,
                             stop=stop,gender="F")
                elif gender == "male":
                    diarization.append(show=show, cluster=loc, start=start,
                             stop=stop,gender="M")
                else:
                    diarization.append(show=show, cluster=loc, start=start,
                             stop=stop)
        except Exception as e:
            logging.error(sys.exc_info()[0])
            logging.error(line)
        fic.close()
        return diarization

    @classmethod
    def read_mdtm(cls, filename, normalize_cluster=False, encoding="utf8"):
        """
        Read a MDTM file
        :param filename: the str input filename
        :param normalize_cluster: normalize the cluster by removing upper case
        and accents
        :return: a diarization object
        """

        fic = open(filename, 'r', encoding=encoding)
        diarization = Diar()
        if not diarization._attributes.exist('gender'):
            diarization.add_attribut(new_attribut='gender', default='U')
        for line in fic:
            line = line.strip()
            line = re.sub('\s+',' ',line)
            logging.debug(line)
            if line.startswith('#') or line.startswith(';;'):
                continue
            # split line into fields
            show, tmp, start_str, length, t, score, gender, cluster = line.split()
            start = int(round(float(start_str)*100, 0))
            stop = start+int(round(float(length)*100, 0))
            if normalize_cluster:
                cluster = str2str_normalize(cluster)
            # print(show, tmp, start, length, gender, channel, env, speaker)
            diarization.append(show=show, cluster=cluster, start=start,
                             stop=stop, gender=gender)
        fic.close()
        return diarization

    @classmethod
    def read_uem(cls, filename, encoding="utf8"):
        """
        Read a UEM file
        :param filename: the str input filename
        :return: a diarization object
        """
        fic = open(filename, 'r', encoding=encoding)
        diarization = Diar()
        if not diarization._attributes.exist('gender'):
            diarization.add_attribut(new_attribut='gender', default='U')
        try:
            name = "uem"
            for line in fic:
                line = re.sub('\s+',' ',line)
                line = line.strip()
                # logging.debug(line)
                if line.startswith('#') or line.startswith(';;'):
                    continue
                # split line into fields
                show, tmp, start_str, stop_str = line.split()
                start = int(round(float(start_str)*100, 0))
                stop = int(round(float(stop_str)*100, 0))
                # stop = start+int(round(float(length)*100, 0))
                diarization.append(show=show, cluster=name, start=start, stop=stop)
        except Exception as e:
            logging.error(sys.exc_info()[0])
            logging.error(line)
        fic.close()
        return diarization

    @classmethod
    def read_rttm(cls, filename, normalize_cluster=False, encoding="utf8"):
        """
        Read rttm file
        :param filename: str input filename
        :param normalize_cluster: normalize the cluster by removing upper case and accents
        :return: a diarization object
        """
        fic = open(filename, 'r', encoding=encoding)
        diarization = Diar()
        if not diarization._attributes.exist('gender'):
            diarization.add_attribut(new_attribut='gender', default='U')
        try:
            for line in fic:
                line = re.sub('\s+',' ',line)
                line = line.strip()
                if line.startswith('#') or line.startswith(';;'):
                    continue
                # split line into fields
                spk, show, tmp0, start_str, length, tmp1, tmp2, cluster, tmp3 = line.split()
                if spk == "SPEAKER":
                    start = int(round(float(start_str)*100, 0))
                    stop = start+int(round(float(length)*100, 0))
                    if normalize_cluster:
                        cluster = str2str_normalize(cluster)
                    diarization.append(show=show, cluster=cluster, start=start, stop=stop)
        except Exception as e:
            logging.error(sys.exc_info()[0])
            logging.error(line)
        fic.close()
        return diarization

    @classmethod
    def to_string_seg(cls, diar):
        """
        transform a diarization into a string
        :param diar: a diarization
        :return: a string
        """
        lst = []
        for segment in diar:
            gender = 'U'
            if diar._attributes.exist('gender'):
                gender = segment['gender']
            env = 'U'
            if diar._attributes.exist('env'):
                env = segment['env']
            channel = 'U'
            if diar._attributes.exist('channel'):
                channel = segment['channel']
            lst.append('{:s} 1 {:d} {:d} {:s} {:s} {:s} {:s}\n'.format(
                segment['show'], segment['start'], segment['stop'] - segment['start'], gender,
                channel, env, segment['cluster']))
        return lst

    @classmethod
    def intersection(cls, diarization1, diarization2):
        """
        Compute the intersection between two diarization
        :param diarization1: first diarization
        :param diarization2: second diarization
        :return: a diarization object
        """
        diarization = Diar()
        for segment1 in diarization1:
            for segment2 in diarization2:
                inter = Segment.intersection(segment1, segment2)
                if inter is not None :
                    diarization.append_seg(inter)
        return diarization

    @classmethod
    def write_seg(cls, filename, diarization):
        """
        Write diarization to a segmentation file
        :param filename: the str output filename
        :param diarization: the diarization to write

        """
        diarization.sort(['show', 'start'])
        fic = open(filename, 'w', encoding="utf8")
        for line in Diar.to_string_seg(diarization):
            fic.write(line)
        fic.close()

    @classmethod
    def write_lbl(cls, diarization, label_dir='', label_file_extension='.lbl'):
        """
        Write diarization to label file
        :param diarization: the diarization to write
        :param label_dir: the string directory of the ouput filename
        :param label_file_extension: the string extension of the output filename

        """
        diarization.sort(['show', 'start'])
        old_show = ''
        fic = None
        for segment in diarization:
            if old_show != segment['show']:
                if fic is not None:
                    fic.close()
                filename = os.path.join(label_dir, segment['show']
                                        + label_file_extension)
                fic = open(filename, 'w')
                old_show = segment['show']
            fic.write('{:d} {:d} {:s}\n'.format(
                segment['start'], segment['stop'], segment['cluster']))
        fic.close()

class Segment(list):
    """
    Class to store the segment informations.


    :attr _attributes: is the list of attribut names
    :attr data: the data associated to each attribut
    """
    def __init__(self, data, attributes):
        """
        Called after the instance has been created (by __new__()), but before it is returned to the caller.
        :param data: copy the row data
        :param attributes: the names of the attributs

        """
        list.__init__(self)
        self._attributes = attributes
        for item in data:
            self.append(item)

    def _get_attr(self, attr_name):
        """
        Called to implement evaluation of self[attr_name].
        :param attr_name: a string
        :return: the value
        """
        return self[self._attributes[attr_name]]

    def _set_attr(self, attr_name, value):
        """
        Called to implement assignment to self[attr_name].

        :param attr_name: a str
        :param value: the value to set

        """
        self[self._attributes[attr_name]] = value

    def __getitem__(self, index):
        """
        Called to implement evaluation of self[index].
        :param index: a int
        :return: the value
        """

        if isinstance(index, str):
            return self._get_attr(index)
        else:
            return list.__getitem__(self, index)

    def __setitem__(self, index, value):
        """
        Called to implement assignment to self[index].
        :param index: a int
        :param value: the value to set
        :return: the item
        """
        if isinstance(index, str):
            return self._set_attr(index, value)
        else:
            return list.__setitem__(self, index, value)

    def __eq__(self, segment): # real signature unknown
        if segment is not None:
            l = len(segment)
            if l != len(self):
                return False

            for i in range(l):
                if self[i] != segment[i]:
                    return False
            return True
        else:
            return False

    def __ne__(self, segment): # real signature unknown
        return not self.__eq__(segment)

    def duration(self):
        """
        :return: the duration of the segment
        """

        return self['stop'] - self['start']

    def seg_features(self, features):
        """
        Given a FeatureServer, returns a list of feature index corresponding to
        the segment.

        :param features: a FeatureServer
        :return: a list of int
        """
        return features[self['start']:self['stop'], :]

    @classmethod
    def gap(cls, segment1, segment2):
        """
        Returns the inter segment gap between 2 segments.
        :param segment1: a Segment object
        :param segment2: a Segment object
        :return: a Segment object

        Examples
        --------
        >>> from s4d.diarization import Diar, Segment
        >>> diarization=Diar()
        >>> diarization.append(show='empty', start=0, stop=100, cluster='spk1')
        >>> diarization.append(show='empty', start=50, stop=150, cluster='spk2')
        >>> s = Segment.intersection(diarization[0], diarization[1])
        >>> s
        ['empty', 'spk1', 'speaker', 100, 50]
        >>> s.duration()
        - 50
        >>> diarization.append(show='empty', start=200, stop=250, cluster='spk1')
        >>> Segment.gap(diarization[0], diarization[2])
        ['empty', 'spk1', 'speaker', 100, 200]


        """
        if segment1['show'] != segment2['show']:
            raise Exception('not the same show')
        segment = Segment(segment1, segment1._attributes)
        segment['start'] = segment1['stop']
        segment['stop'] = segment2['start']
        return segment
    @classmethod
    def split(cls, segment1, segment2):
        inter = cls.intersection(segment1, segment2)
        diff = cls.diff(segment1, segment2)

    @classmethod
    def intersection(cls, segment1, segment2):
        """
        Intersection between 2 segments. Return None if the intersection is empty.
        :param segment1: a Segment object
        :param segment2: a Segment object
        :return: a Segment object

        Examples
        --------
        >>> from s4d.diarization import Diar, Segment
        >>> diarization=Diar()
        >>> diarization.append(show='empty', start=0, stop=100, cluster='spk1')
        >>> diarization.append(show='empty', start=50, stop=150, cluster='spk2')
        >>> Segment.intersection(diarization[0], diarization[1])
        ['empty', 'spk1 / spk2', 'speaker', 50, 100]
        >>> diarization.append(show='empty', start=50, stop=75, cluster='spk1')
        >>> Segment.intersection(diarization[0], diarization[2])
        ['empty', 'spk1 / spk1', 'speaker', 50, 75]
        >>> diarization.append(show='empty', start=200, stop=250, cluster='spk1')
        >>> s = Segment.intersection(diarization[0], diarization[3])
        >>> s is None

        True

        """
        if segment1['show'] != segment2['show']:
            raise Exception(
                'not the same show ' + segment1['show'] + ' != ' + segment2['show'])
        segment = Segment(segment1, segment1._attributes)
        segment['cluster'] += '##' + segment2['cluster']
        segment['start'] = max(segment1['start'], segment2['start'])
        segment['stop'] = min(segment1['stop'], segment2['stop'])
        if segment.duration() > 0:
            return segment
        return None

    @classmethod
    def diff(cls, segment1, segment2):
        """
        The difference between the two segments. Returns one or two segments and the
        source of the new segment: 1 means segment1, 2 means segment2.

        :param segment1: a Segment object
        :param segment2: a Segment object
        :return: a list of segments
        """
        lst_row = list()

        if segment1['show'] != segment2['show']:
            return lst_row
        start = min(segment1['start'], segment2['start'])
        stop = max(segment1['start'], segment2['start'])
        if stop - start > 0:
            if start == segment1['start']:
                segment = copy.deepcopy(segment1)
            else:
                segment = copy.deepcopy(segment2)
            segment['stop'] = stop
            lst_row.append(segment)

        start = min(segment1['stop'], segment2['stop'])
        stop = max(segment1['stop'], segment2['stop'])
        if stop - start > 0:
            if stop == segment1['stop']:
                segment = copy.deepcopy(segment1)
            else:
                segment = copy.deepcopy(segment2)
            segment['start'] = start
            lst_row.append(segment)
        return lst_row

    @classmethod
    def union(cls, segment1, segment2):
        """
        Union between 2 segments.
        :param segment1: a Segment object
        :param segment2: a Segment object
        :return: a Segment object

        Examples
        --------
        >>> from s4d.diarization import Diar, Segment
        >>> diarization=Diar()
        >>> diarization.append(show='empty', start=0, stop=100, cluster='spk1')
        >>> diarization.append(show='empty', start=50, stop=150, cluster='spk2')
        >>> Segment.union(diarization[0], diarization[1])
        ['empty', 'spk1', 'speaker', 0, 150]
        >>> diarization.append(show='empty', start=50, stop=75, cluster='spk1')
        >>> Segment.union(diarization[0], diarization[2])
        ['empty', 'spk1', 'speaker', 0, 100]
        >>> diarization.append(show='empty', start=200, stop=250, cluster='spk1')
        >>> Segment.union(diarization[0], diarization[3])

        ['empty', 'spk1', 'speaker', 0, 250]

        True

        """
        if segment1['show'] != segment2['show']:
            raise Exception('not the same show ' +
                            segment1['show'] + ' != ' + segment2['show'])
        segment = Segment(segment1, segment1._attributes)
        segment['start'] = min(segment1['start'], segment2['start'])
        segment['stop'] = max(segment1['stop'], segment2['stop'])
        return segment


class Index(dict):
    """Implementation of perl's autovivification feature.
    Thanks to http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
    """

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class AttributeNames:
    """
    Class AttributeNames defines a list of column names
    """
    def __init__(self):
        self.names = dict
        self.defaults = list

    def index_of(self, name):
        """
        :param name: the speaker of the column
        :return: the position (a int) of the speaker
        """
        return self.names[name]

    def exist(self, name):
        """
        Test the existing value of speaker in the list of column names
        :param name: the speaker of the column
        :return: a boolean
        """
        return name in self.names

    def initialize(self, names, defaults):
        """
        Initialaze AttributeNames object
        :param names: a list of column names
        :param defaults: the list of default values

        """
        self.names = names
        self.defaults = defaults

    def __getitem__(self, index):
        """
        Called to implement evaluation of self[index].
        :param index: a int
        :return: the value
        """
        return self.names[index]

    def __setitem__(self, index, value):
        """
        Called to implement assignment to self[index].
        :param index: a int
        :param value: the value to set
        :return: the item
        """
        return self.names.__setitem__(index, value)

    def __iter__(self):
        """
        This method is called when an iterator is required for a container.
        :return: a iterator on the column names
        """
        return self.names.__iter__()

    def __len__(self):
        """
        Get the number of column names
        :return:
        """
        return self.defaults.__len__()

    def sorted(self):
        """
        sort the column names
        :return:
        """
        return sorted(self.names.items(), key=lambda x: x[1])

    def add(self, name, default=''):
        """
        a a column speaker
        :param name: a str
        :param default: the default value

        """
        if name in self.names:
            raise Exception('This attribut exits : ')
        else:
            self[name] = len(self.defaults)
            self.defaults.append(default)

    def delete(self, name):
        """
        remove a column and the default value  given it column speaker
        :param name:

        """
        if name not in self.names:
            raise Exception("This attribut don't exits : " + name)
        else:
            i = self[name]
            del self.defaults[i]
            del self.names[name]
            for k in self.names:
                if self.names[k] > i:
                    self.names[k] -= 1


def rolling_window(a, window):
    """
    Make an ndarray with a rolling window of the last dimension

    Examples
    --------
    >>> x=numpy.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:
    >>> numpy.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    :param a: Array to add rolling window to
    :param window: Size of rolling window
    :return: Array that is a view of the original array with a added dimension
    of size w.
    """

    if window < 1:
        raise (ValueError, "`window` must be at least 1.")
    if window > a.shape[-1]:
        raise (ValueError, "`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



