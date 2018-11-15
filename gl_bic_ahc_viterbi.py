#-*- coding:utf-8 -*-
from s4d.utils import *
from s4d.diar import Diar
from s4d import viterbi, segmentation
from sidekit import features_server
from s4d.clustering import hac_bic
from sidekit.sidekit_io import init_logging
from s4d.gui.dendrogram import plot_dendrogram
import h5py
import speakin_ivector
import numpy as np
from pydub import AudioSegment
from collections import OrderedDict
import speakin_voice_feats
import os



_FRAME_SHIFT = 10
_FRAME_LENGTH = 25
_mfccExtractor = speakin_voice_feats.genMfccExtractor(
    allow_downsample=True, sample_frequency=8000, frame_length=_FRAME_LENGTH, frame_shift=_FRAME_SHIFT, high_freq=3700, low_freq=20)
_deltaExtractor = speakin_voice_feats.createDeltaFeatures()

def _paramSt():
    win_size = 100
    thr_l = 0
    thr_h = 1
    thr_vit = -80
    return win_size, thr_l, thr_h, thr_vit

def _setOut(outPath):
    wdir = outPath
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    return wdir

def _getMfcc(wavFile, wdir, save_all):
    wavName = wavFile[:-4]
    if save_all:
        fe = get_feature_extractor(wavFile, type_feature_extractor='sid8k')
        mfcc_filename = os.path.join(wdir, wavName + '.mfcc.h5')
        fe.save(wavName, input_audio_filename=wavFile, output_feature_filename=mfcc_filename)
        fs = get_feature_server(mfcc_filename, feature_server_type='sid8k')
    else:
        fs = get_feature_server(wavFile, feature_server_type='sid8k')
    cep, vad = fs.load(wavName)
    return cep

def _speakinMfcc(audioPath):
    rawAudio = AudioSegment.from_wav(audioPath)
    if rawAudio.frame_rate != 8000:
        raise Exception("wrong sample rate")
    if rawAudio.sample_width != 2:
        raise Exception("wrong sample width")
    if rawAudio.channels != 1:
        raise Exception("wrong channel count")
    mfccFeats = speakin_voice_feats.extractFeats(_mfccExtractor, audioPath)
    mfccFeats = speakin_voice_feats.computeDelta(_deltaExtractor, mfccFeats)
    mfccFeats = np.array(mfccFeats)

    return rawAudio, mfccFeats

def _initialSegmentation(mfcc, wavFile,
                         save_all, wdir):
    wavName = wavFile[:-4]
    init_diar = segmentation.init_seg(mfcc, wavName)
    if save_all:
        init_filename = os.path.join(wdir, wavName + '.i.seg')
        Diar.write_seg(init_filename, init_diar)
    return init_diar

def _gaussDiverSegmentation(mfcc, wavFile,
                            init_diar, win_size, wdir, save_all):
    wavName = wavFile[:-4]
    seg_diar = segmentation.segmentation(mfcc, init_diar, win_size)
    if save_all:
        seg_filename = os.path.join(wdir, wavName + '.s.seg')
        Diar.write_seg(seg_filename, seg_diar)
    return seg_diar

def _linearBic(mfcc, seg_diar,
               thr_l, wavFile, wdir, save_all):
    wavName = wavFile[:-4]
    bicl_diar = segmentation.bic_linear(mfcc, seg_diar, thr_l, sr=False)
    if save_all:
        bicl_filename = os.path.join(wdir, wavName + '.l.seg')
        Diar.write_seg(bicl_filename, bicl_diar)
    return bicl_diar

def _bicAhc(mfcc, bicl_diar,
            thr_h, wavFile, wdir, save_all):
    wavName = wavFile[:-4]
    bic = hac_bic.HAC_BIC(mfcc, bicl_diar, thr_h, sr=False)
    bich_diar = bic.perform(to_the_end=True)
    if save_all:
        bichac_filename = os.path.join(wdir, wavName + '.h.seg')
        Diar.write_seg(bichac_filename, bich_diar)
    #link, data = plot_dendrogram(bic.merge, 0)
    return bich_diar

def _viterbiDecode(mfcc, bich_diar, thr_vit, wavFile, wdir, save_all):
    wavName = wavFile[:-4]
    vit_diar = viterbi.viterbi_decoding(mfcc, bich_diar, thr_vit)
    if save_all:
        vit_filename = os.path.join(wdir, wavName + '.d.seg')
        Diar.write_seg(vit_filename, vit_diar)
    return vit_diar

def _outputWave(wavFile, vit_diar, outPath):
    finalList = []
    spkMap = OrderedDict()
    rawAudio = AudioSegment.from_wav(wavFile)
    for info in vit_diar:
        spk, start, stop = info[1], info[3], info[4]
        if spk not in spkMap.keys():
            spkMap[spk] = [(start,stop)]
        else:
            spkMap[spk].append((start,stop))
    for spk, segList in spkMap.items():
        finalList.append(segList)
        print(spk, segList)
        outAudio = None
        for start, end in segList:
            start = start * 10
            end = end * 10
            if outAudio is None:
                outAudio = rawAudio[start:end]
            else:
                outAudio = outAudio + rawAudio[start:end]
        if outAudio:
            outFile = "{}/{}.wav".format(outPath, spk)
            outAudio.export(outFile, format='wav')
    return finalList

def _diarizationProcess(wavFile, mfccMethod, outPath):


    save_all = True
    win_size, thr_l, thr_h, thr_vit = _paramSt()
    wdir = _setOut(outPath)
    mfccMethod = mfccMethod.upper()
    # two methods for getting audio mfcc, one from original s4d, and the other from Kaldi
    if mfccMethod == 'S4D':
        print('Using s4d mfcc extraction')
        mfcc = _getMfcc(wavFile, wdir, save_all)
        print('Testing for mfcc extraction', mfcc.shape)

    elif mfccMethod == 'KALDI':
        print('Using Kaldi mfcc extraction')
        rawAudio, mfcc = _speakinMfcc(wavFile)
        print('Testing for mfcc extraction', mfcc.shape)
    else:
        print('Need to type a correct mfcc extraction method ')
        print('Useage: {} [wavFile] [mfccMethod] [outPath]'.format(
            sys.argv[0]))
        print('mfccMethod: "s4d" or "kaldi"')
        sys.exit(0)

    init_diar = _initialSegmentation(mfcc, wavFile, save_all, wdir)
    seg_diar = _gaussDiverSegmentation(mfcc, wavFile, init_diar, win_size, wdir, save_all)
    bicl_diar = _linearBic(mfcc, seg_diar, thr_l, wavFile, wdir, save_all)
    bich_diar = _bicAhc(mfcc, bicl_diar, thr_h, wavFile, wdir, save_all)
    vit_diar = _viterbiDecode(mfcc, bich_diar, thr_vit, wavFile, wdir, save_all)
    return vit_diar

# def s4d_diarization(wavFile, mfccMethod, win_size, thr_l, thr_h, thr_vit):
#     vit_diar = _diarizationProcess(wavFile, mfccMethod, outPath, win_size, thr_l, thr_h, thr_vit)
#     finalList = _outputWave(wavFile, vit_diar, outPath)      #generate output wave file after diarization


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Useage: {} [wavFile] [mfccMethod] [outPath]'.format(
            sys.argv[0]))
        print('mfccMethod: kaldi : mfcc extraction from kaldi, s4d: mfcc extraction from s4d')
        sys.exit(0)
    wavFile = sys.argv[1]
    mfccMethod = sys.argv[2]
    outPath = sys.argv[3]
    vit_diar = _diarizationProcess(wavFile, mfccMethod, outPath)
    finalList = _outputWave(wavFile, vit_diar, outPath)      #generate output wave file after diarization
    print(finalList)
