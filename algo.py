#-*- coding:utf-8 -*-
from s4d.utils import *
from s4d import viterbi, segmentation
from s4d.clustering import hac_bic
import numpy as np
from pydub import AudioSegment
from collections import OrderedDict
import speakin_voice_feats
from io import BytesIO

_FRAME_SHIFT = 10
_FRAME_LENGTH = 25
_mfccExtractor = speakin_voice_feats.genMfccExtractor(
    allow_downsample=True, sample_frequency=8000, frame_length=_FRAME_LENGTH, frame_shift=_FRAME_SHIFT, high_freq=3700, low_freq=20)
_deltaExtractor = speakin_voice_feats.createDeltaFeatures()


def _getMfcc(wavFile):
    wavName = wavFile[:-4]
    fs = get_feature_server(wavFile, feature_server_type='sid8k')
    cep, vad = fs.load(wavName)
    return cep


def _speakinMfcc(audioPath):
    mfccFeats = speakin_voice_feats.extractFeats(_mfccExtractor, audioPath)
    mfccFeats = speakin_voice_feats.computeDelta(_deltaExtractor, mfccFeats)
    mfccFeats = np.array(mfccFeats)
    print(mfccFeats)
    return mfccFeats


def _initialSegmentation(mfcc, wavFile):
    return segmentation.init_seg(mfcc, wavFile[:-4])


def _gaussDiverSegmentation(mfcc, init_diar, win_size):
    return segmentation.segmentation(mfcc, init_diar, win_size)


def _linearBic(mfcc, seg_diar, thr_l):
    return segmentation.bic_linear(mfcc, seg_diar, thr_l, sr=False)


def _bicAhc(mfcc, bicl_diar, thr_h):
    bic = hac_bic.HAC_BIC(mfcc, bicl_diar, thr_h, sr=False)
    bich_diar = bic.perform(to_the_end=True)
    return bich_diar

def _viterbiDecode(mfcc, bich_diar, thr_vit):
    return viterbi.viterbi_decoding(mfcc, bich_diar, thr_vit)


def _diarizationProcess(wavFile, mfccMethod, win_size, thr_l, thr_h, thr_vit):
    finalList = []
    spkMap = OrderedDict()

    win_size, thr_l, thr_h, thr_vit = win_size, thr_l, thr_h, thr_vit
    mfccMethod = mfccMethod.upper()
    # two methods for getting audio mfcc, one from original s4d, and the other from Kaldi
    if mfccMethod == 'S4D':
        print('Using s4d mfcc extraction')
        mfcc = _getMfcc(wavFile)
        print('Testing for mfcc extraction', mfcc.shape)

    elif mfccMethod == 'KALDI':
        print('Using Kaldi mfcc extraction')
        mfcc = _speakinMfcc(wavFile)
        print('Testing for mfcc extraction', mfcc.shape)
    else:
        print('Need to type a correct mfcc extraction method ')
        print('Useage: {} [wavFile] [mfccMethod] [outPath]'.format(
            sys.argv[0]))
        print('mfccMethod: "s4d" or "kaldi"')
        sys.exit(0)

    init_diar = _initialSegmentation(mfcc, wavFile,)
    seg_diar = _gaussDiverSegmentation(mfcc, init_diar, win_size)
    bicl_diar = _linearBic(mfcc, seg_diar, thr_l)
    bich_diar = _bicAhc(mfcc, bicl_diar, thr_h)
    vit_diar = _viterbiDecode(mfcc, bich_diar, thr_vit)

    for info in vit_diar:
        spk, start, stop = info[1], info[3], info[4]
        if spk not in spkMap.keys():
            spkMap[spk] = [(start*10, stop*10)]
        else:
            spkMap[spk].append((start*10, stop*10))
    for spk, segList in spkMap.items():
        finalList.append(segList)
    return spkMap

def _getSegAudioData(segAudio):
    f = BytesIO()
    segAudio.export(f, format="wav")
    return f.getvalue()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Useage: {} [wavFile] [mfccMethod] [outPath]'.format(
            sys.argv[0]))
        print('mfccMethod: kaldi : mfcc extraction from kaldi, s4d: mfcc extraction from s4d')
        sys.exit(0)
    wavFile = sys.argv[1]
    mfccMethod = sys.argv[2]
    win_size = 100
    thr_l = 0
    thr_h = 1
    thr_vit = -100

    spkMap = _diarizationProcess(wavFile, mfccMethod, win_size, thr_l, thr_h, thr_vit)
    print(spkMap)
    res = {}
    for key, value in spkMap.items():
        print(key, value)
        name = str(key)
        times = np.array(value).reshape(-1)
        print(times)
        dim_2 = len(times) // 2
    print(res)