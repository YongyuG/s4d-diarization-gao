#!/usr/bin/env python
#-*- coding:utf-8 -*-


import numpy as np
import speakin_base_algo_agent as algo_agent
from speakin_base_algo_agent.algo_types import *
from io import BytesIO
from algo import _diarizationProcess, _speakinMfcc
import os
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'

class Configures(object):
    algoName = "speaker_diarization"
    algoDesc = "speaker_diarization_BIC_clustering"
    moduleName = "speaker_diarization_16k"
    algoVersion = 1
    moduleVersion = 1



class AlgoAgent(algo_agent.BaseAlgoAgent):
    def __init__(self, args=[]):
        pass


    def getAlgoInfo(self):
        methodList = [
        MethodInfo(methodName="speaker_diarization_S4D", methodDesc="基于S4D的人声分离",
                       reqDefList=[
                           DataValueItemDef(itemName="file", itemDesc="文件",
                                            itemType=ValueType.VALUE_FILE_ID),
                           DataValueItemDef(itemName="mfccMethod", itemDesc="0kaldi,1表示kemans,2表示AHC,默认为0",
                                            itemType=ValueType.VALUE_STRING),

                       ],
                       respDefList=[DataValueItemDef(itemName="voice{num}", itemDesc="声音区间, num表示第几个人",
                                                     itemType=ValueType.VALUE_TENSOR),
                                    # DataValueItemDef(itemName="bobao",
                                    #                  itemDesc="播报区间",
                                    #                  itemType=ValueType.VALUE_TENSOR),
                                    # DataValueItemDef(itemName="mangyin",
                                    #                  itemDesc="忙音区间",
                                    #                  itemType=ValueType.VALUE_TENSOR),
                                    # DataValueItemDef(itemName="music",
                                    #                  itemDesc="铃声区间",
                                    #                  itemType=ValueType.VALUE_TENSOR),
                                    ]),
        ]
        return AlgoInfo(algoName=Configures.algoName, algoVersion=Configures.algoVersion, algoDesc=Configures.algoDesc,
                        moduleName=Configures.moduleName, moduleVersion=Configures.moduleVersion, methodList=methodList)

    def call(self, methodName, req):
        if "speaker_diarization_S4D" != methodName:
            return None
        if "file" not in req.fileMap:
            return None
        data = req.fileMap["file"]
        mfccMethod = 'kaldi'
        mfccMethodMap = {"0": "kaldi", "1": "s4d"}
        win_size = 100
        thr_l = 0
        thr_h = 1
        thr_vit = -100
        if "mfccMethod" in req.intMap.keys():
            index =  req.intMap["mfccMethod"]
            mfccMethod = mfccMethodMap[index]
        seg_point = _diarizationProcess(data, mfccMethod, win_size, thr_l, thr_h, thr_vit)
        res = {}
        for key, value in seg_point.items():
            name = "voice" + str(key)
            times = np.array(value).reshape(-1)
            print(times.dtype)
            dim_2 = len(times) // 2
            t = Tensor([dim_2, 2], times.tolist())
            res[name] = t
        return DataValue(tensorMap=res)


args = algo_agent.parseCommandLine()
algo_agent.runServer(AlgoAgent(), args)

