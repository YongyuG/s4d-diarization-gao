# coding: utf8
from __future__ import unicode_literals

import logging
import numbers
import copy
import collections
import numpy as np
import pandas as pd
import pyannote.metrics.segmentation as pyaseg
import pyannote.metrics.diarization as pyadiar
import pyannote.core as pyacore
from s4d.scoring import DER
from s4d.diar import Diar, Segment

# Returns a diar object by adjusting the boundaries according both a diar and a tolerance
## WARNING: The boundary matching rests on the nearest distance. In any case, it doesn't take into consideration the labels
## tolerance: In centiseconds
def adjustBoundAccordingToDiarAndTolerance(diar,diarBasis,diarUem=None,tolerance=25):
    assert isinstance(diar,Diar) and isinstance(diarBasis,Diar) and ((isinstance(diarUem,Diar) and len(diarOverlapArea(diarUem))==0) or diarUem is None) and isinstance(tolerance,numbers.Number)
    basis=boundHypToChange(diar,diarBasis,diarUem,False,tolerance)
    basisI={v: k for k, v in basis.items()}
    dOut=copy.deepcopy(diar)
    for i in dOut:
        if i['start'] in basisI:
            i['start']=basisI[i['start']]
        if i['stop'] in basisI:
            i['stop']=basisI[i['stop']]
    return dOut

# Returns a diar object with a new column detailing the overlapped segments
def advancedOverlapDiar(diar):
    assert isinstance(diar,Diar)
    out_diar=diarOverlapArea(diar)
    out_diar.add_attribut("OverlappedSegments",None)
    for i in out_diar:
        listTmp=list()
        for j in diar:
            if Segment.intersection(i,j) is not None:
                listTmp.append(copy.deepcopy(j))
        i["OverlappedSegments"]=listTmp

    out_diar_tmp=copy.deepcopy(diar)
    out_diar_tmp.add_attribut("OverlappedSegments",None)
    for i in out_diar:
        out_diar_tmp=releaseFramesFromSegment(i,out_diar_tmp)
    out_diar.append_diar(out_diar_tmp)
    out_diar.sort()
    return out_diar

# Returns a dict object with an automaton which only corrects the assignment errors
## WARNING: The diarizations in parameter have to be with no overlapped segment. Put them apart
## WARNING: The automaton follows the temporal order
## tolerance: In centiseconds
## diarFinal__clusterToDeleteAccordingToDiarRef: List of clusters to delete in the diarFinal only
def automatonAssignment(diarHyp,diarRef,diarUem=None,tolerance=0,diarFinal__clusterToDeleteAccordingToDiarRef=list()):
    assert isinstance(diarHyp,Diar) and (diarUem is None or isinstance(diarUem,Diar)) and isinstance(diarRef,Diar) and isinstance(tolerance,numbers.Number) and isinstance(diarFinal__clusterToDeleteAccordingToDiarRef,list)
    for u in diarFinal__clusterToDeleteAccordingToDiarRef:
        assert isinstance(u,str)

    actionsAssignmentHumanCorrection=collections.OrderedDict()
    actionsAssignmentCreate=list()
    actionsAssignmentChange=list()
    actionsAssignmentNothing=list()
    actionsAssignmentCreateBis=list()
    actionsAssignmentHumanCorrection["Create"]=actionsAssignmentCreate
    actionsAssignmentHumanCorrection["Change"]=actionsAssignmentChange
    actionsAssignmentHumanCorrection["Nothing"]=actionsAssignmentNothing
    dictionary=dict()

    actionsIncrementalAssignmentHumanCorrection=collections.OrderedDict()
    actionsIncrementalAssignmentCreate=list()
    actionsIncrementalAssignmentChange=list()
    actionsIncrementalAssignmentNothing=list()
    actionsIncrementalAssignmentHumanCorrection["Create"]=actionsIncrementalAssignmentCreate
    actionsIncrementalAssignmentHumanCorrection["Change"]=actionsIncrementalAssignmentChange
    actionsIncrementalAssignmentHumanCorrection["Nothing"]=actionsIncrementalAssignmentNothing

    diarIncremental=dict()

    idxIncremental=dict()

    if diarUem is not None:
        diarRef=releaseFramesAccordingToDiar(diarRef,diarUem)
        diarHyp=releaseFramesAccordingToDiar(diarHyp,diarUem)

    diarRaw=Diar()
    diarRaw.append(start=min(diarRef.unique('start')+diarHyp.unique('start')),stop=max(diarRef.unique('stop')+diarHyp.unique('stop')))
    diarRef=copy.deepcopy(diarRef)
    diarHyp=copy.deepcopy(diarHyp)
    diarRef.sort()
    diarHyp.sort()
    tolerance=abs(tolerance)     

    assert len(diarOverlapArea(diarRef))==0, "Error: diarRef parameter have some overlapped segments.\nReason: No overlap segment allowed.\nSolution: Please put them apart.\n"
    assert len(diarOverlapArea(diarHyp))==0, "Error: diarHyp parameter have some overlapped segments.\nReason: No overlap segment allowed.\nSolution: Please put them apart.\n"

    actionsIncrementalAssignmentCreateTurn=list()
    actionsIncrementalAssignmentChangeTurn=list()
    actionsIncrementalAssignmentNothingTurn=list()

    # To avoid to create clusters with the same id
    cpt=0

    for j in diarHyp:

        idxIncremental[len(idxIncremental)]=(j['start'],j['stop'])
        valueRefTmp=None
        diarTmp=Diar()
        diarTmp.append_seg(j)
        match=matchingSegmentsFromSegment(j,diarRef)

        bestMatchValue=None
        bestMatch=None
        if len(match)!=0:
            for x in match:
                if bestMatchValue is None:
                    bestMatchValue=match[x]
                    bestMatch=x
                elif bestMatchValue.duration()<match[x].duration():
                    bestMatchValue=match[x]
                    bestMatch=x
        fakeTmp=Diar.intersection(diarTmp,diarRef)
        if fakeTmp is not None:
            fakeDuration=j.duration()-fakeTmp.duration()
        else:
            fakeDuration=0

        if len(match)!=0:
            if bestMatchValue.duration() < fakeDuration:
                valueRefTmp='speakerManualFake'
            else:
                valueRefTmp=bestMatch
        else:
            valueRefTmp='speakerManualFake'

        keep=False
        if valueRefTmp!='speakerManualFake':
            for y in diarRef:
                if Segment.intersection(y,j) is not None and segmentExistAccordingToTolerance(y,tolerance):
                    keep=True
                    break
        else:
            diarRefFake=copy.deepcopy(diarRaw)
            if diarUem is not None:
                diarRefFake=releaseFramesAccordingToDiar(diar=diarRefFake,basisDiar=diarUem)
            diarRefFake=releaseFramesAccordingToDiar(diar=diarRefFake,basisDiar=diarRef)
            
            for y in diarRefFake:
                y['show']==j['show']
                if Segment.intersection(y,j) is not None and segmentExistAccordingToTolerance(y,tolerance):
                    keep=True
                    break

        if not keep:
            diarHyp=dropSegment(j,diarHyp)
        else:
            applyChange=False
            if valueRefTmp == "speakerManualFake":
                speakerName="speakerManualFake"
            else:
                speakerName="speakerManual"
            if valueRefTmp not in dictionary:
                if j['cluster'] in actionsAssignmentCreateBis:
                    dictionary[valueRefTmp]=speakerName+str(cpt+1)
                    actionsAssignmentCreateBis.append(speakerName+str(cpt+1))
                    actionsAssignmentCreate.append([copy.deepcopy(valueRefTmp),speakerName+str(cpt+1),copy.deepcopy(j)])
                    actionsIncrementalAssignmentCreateTurn.append([copy.deepcopy(valueRefTmp),speakerName+str(cpt+1),copy.deepcopy(j)])
                    applyChange=True
                    cpt+=1
                else:
                    dictionary[valueRefTmp]=copy.deepcopy(j['cluster'])
                    actionsAssignmentCreateBis.append(copy.deepcopy(j['cluster']))
                    actionsAssignmentCreate.append(copy.deepcopy([valueRefTmp,j['cluster'],copy.deepcopy(j)]))
                    actionsIncrementalAssignmentCreateTurn.append(copy.deepcopy([valueRefTmp,j['cluster'],copy.deepcopy(j)]))
            else:
                if j['cluster'] == dictionary[valueRefTmp]:
                    actionsAssignmentNothing.append(copy.deepcopy(j))
                    actionsIncrementalAssignmentNothingTurn.append(copy.deepcopy(j))
                else:
                    actionsAssignmentChange.append(copy.deepcopy([dictionary[valueRefTmp],j]))
                    actionsIncrementalAssignmentChangeTurn.append(copy.deepcopy([dictionary[valueRefTmp],j]))
                    applyChange=True
            if applyChange:
                # Updates the diar object for the merges afterward
                segmentTmp=copy.deepcopy(j)
                segmentTmp['cluster']=dictionary[valueRefTmp]
                diarHyp=dropSegment(j,diarHyp)
                diarHyp.append_seg(segmentTmp)
                diarHyp.sort()

        actionsIncrementalAssignmentCreate.append(actionsIncrementalAssignmentCreateTurn)
        actionsIncrementalAssignmentChange.append(actionsIncrementalAssignmentChangeTurn)
        actionsIncrementalAssignmentNothing.append(actionsIncrementalAssignmentNothingTurn)
        actionsIncrementalAssignmentCreateTurn=list()
        actionsIncrementalAssignmentChangeTurn=list()
        actionsIncrementalAssignmentNothingTurn=list()

        # Stores each diar after each human interaction
        diarIncremental[len(diarIncremental)]=(copy.deepcopy(diarHyp))

    # Deletes segments whose the cluster mainly matches with those present in diarFinal__clusterToDeleteAccordingToDiarRef
    for u in diarFinal__clusterToDeleteAccordingToDiarRef:
        if u in dictionary:
            diarHyp=dropCluster(dictionary[u],diarHyp)
    
    rtn=dict()
    rtn['idxIncremental']=idxIncremental
    rtn['diar']=dict()
    rtn['diar']['final']=diarHyp
    rtn['diar']['incremental']=diarIncremental
    rtn['action']=dict()
    rtn['action']['incremental']=actionsIncrementalAssignmentHumanCorrection
    rtn['action']['sum']=actionsAssignmentHumanCorrection

    return rtn

# Returns a dict object with an automaton which only corrects the segmentation errors 
## WARNING: The diarizations in parameter have to be with no overlapped segment. Put them apart
## WARNING: The automaton follows the temporal order
## tolerance: In centiseconds
## diarFinal__clusterToDeleteAccordingToDiarRef: List of clusters to delete in the diarFinal only
## modeNoGap: Drops or not the segment actions (i.e. createSegment & deleteSegment)
## modeNoGap__mergeStrat_BiggestCluster: Whether we merge in the temporal order or first the biggest cluster for a given reference segment (only useful when the modeNoGap is False)
## deleteBoundarySameConsecutiveSpk: Whether we delete a boundary for two consecutive segments with the same speaker
def automatonSegmentation(diarHyp,diarRef,diarUem=None,tolerance=0,modeNoGap=False,modeNoGap__mergeStrat_BiggestCluster=False,diarFinal__clusterToDeleteAccordingToDiarRef=list(),deleteBoundarySameConsecutiveSpk=False):
    assert isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and isinstance(modeNoGap__mergeStrat_BiggestCluster,bool) and isinstance(modeNoGap,bool) and (diarUem is None or isinstance(diarUem,Diar)) and isinstance(tolerance,numbers.Number) and isinstance(diarFinal__clusterToDeleteAccordingToDiarRef,list) and isinstance(deleteBoundarySameConsecutiveSpk,bool)
    for u in diarFinal__clusterToDeleteAccordingToDiarRef:
        assert isinstance(u,str)

    actionsSegmentationHumanCorrection=collections.OrderedDict()
    actionsSegmentationBoundary=collections.OrderedDict()
    actionsSegmentationBoundaryCreate=list()
    actionsSegmentationBoundaryMerge=list()
    # Create Format: [segment, position of the new boundary] -> We have to cut the segment into two parts
    actionsSegmentationBoundary["Create"]=actionsSegmentationBoundaryCreate
    # Merge Format: [segment1, segment2] -> We have to move the segment (the two segments have to have the same attributes)
    actionsSegmentationBoundary["Merge"]=actionsSegmentationBoundaryMerge
    if modeNoGap == False:
        actionsSegmentationSegment=collections.OrderedDict()        
        actionsSegmentationSegmentCreate=list()
        actionsSegmentationSegmentDelete=list()
        # Create Format: [show,cluster,cluster_type, start, end] -> We have to create a new segment
        actionsSegmentationSegment["Create"]=actionsSegmentationSegmentCreate
        # Delete Format: [segment] -> We have to delete a segment
        actionsSegmentationSegment["Delete"]=actionsSegmentationSegmentDelete
    # Nothing Format: [segment] -> Nothing to do, correct segmentation
    actionsSegmentationNothing=list()
    actionsSegmentationHumanCorrection["Boundary"]=actionsSegmentationBoundary
    if modeNoGap == False:
        actionsSegmentationHumanCorrection["Segment"]=actionsSegmentationSegment
    actionsSegmentationHumanCorrection["Nothing"]=actionsSegmentationNothing

    actionsIncrementalSegmentationHumanCorrection=collections.OrderedDict()
    actionsIncrementalSegmentationBoundary=collections.OrderedDict()
    actionsIncrementalSegmentationBoundaryCreate=list()
    actionsIncrementalSegmentationBoundaryMerge=list()
    # Create Format: [segment, position of the new boundary] -> We have to cut the segment into two parts
    actionsIncrementalSegmentationBoundary["Create"]=actionsIncrementalSegmentationBoundaryCreate
    # Merge Format: [segment1, segment2] -> We have to move the segment (the two segments have to have the same attributes)
    actionsIncrementalSegmentationBoundary["Merge"]=actionsIncrementalSegmentationBoundaryMerge
    if modeNoGap == False:
        actionsIncrementalSegmentationSegment=collections.OrderedDict()        
        actionsIncrementalSegmentationSegmentCreate=list()
        actionsIncrementalSegmentationSegmentDelete=list()
        # Create Format: [show,cluster,cluster_type, start, end] -> We have to create a new segment
        actionsIncrementalSegmentationSegment["Create"]=actionsIncrementalSegmentationSegmentCreate
        # Delete Format: [segment] -> We have to delete a segment
        actionsIncrementalSegmentationSegment["Delete"]=actionsIncrementalSegmentationSegmentDelete
    # Nothing Format: [segment] -> Nothing to do. Correct segmentation
    actionsIncrementalSegmentationNothing=list()
    actionsIncrementalSegmentationHumanCorrection["Boundary"]=actionsIncrementalSegmentationBoundary
    if modeNoGap == False:
        actionsIncrementalSegmentationHumanCorrection["Segment"]=actionsIncrementalSegmentationSegment
    actionsIncrementalSegmentationHumanCorrection["Nothing"]=actionsIncrementalSegmentationNothing

    diarIncremental=dict()

    idxIncremental=dict()

    if diarUem is not None:
        diarRef=releaseFramesAccordingToDiar(diarRef,diarUem)
        diarHyp=releaseFramesAccordingToDiar(diarHyp,diarUem)

    diarRaw=Diar()
    diarRaw.append(start=min(diarRef.unique('start')+diarHyp.unique('start')),stop=max(diarRef.unique('stop')+diarHyp.unique('stop')))
    diarRef=copy.deepcopy(diarRef)
    diarHyp=copy.deepcopy(diarHyp)
    showname=diarRef.unique('show')[0]
    diarRef.sort()
    diarHyp.sort()
    tolerance=abs(tolerance)
    if not strictBoundary:
        diarRef.pack()
        diarHyp.pack()     

    assert len(diarOverlapArea(diarRef))==0, "Error: diarRef parameter have some overlapped segments.\nReason: No overlap segment allowed.\nSolution: Please put them apart.\n"
    assert len(diarOverlapArea(diarHyp))==0, "Error: diarHyp parameter have some overlapped segments.\nReason: No overlap segment allowed.\nSolution: Please put them apart.\n"

    actionsIncrementalSegmentationBoundaryCreateTurn=list()
    actionsIncrementalSegmentationBoundaryMergeTurn=list()
    if modeNoGap == False:
        actionsIncrementalSegmentationSegmentCreateTurn=list()
        actionsIncrementalSegmentationSegmentDeleteTurn=list()
    actionsIncrementalSegmentationNothingTurn=list()

    # To avoid to create clusters with the same id
    cpt=0

    for i,valueRef in enumerate(diarRef):
    # WARNING: Each string supposes the start boundary is validate/correct (modified in the previous iteration if need be), that it doesn't overtake the reference segment (works with the tolerance as well)

    # SELECTS ALL THE HYPOTHESIS SEGMENTS BEFORE THE FIRST REFERENCE SEGMENT (means wrong clustered since silence in the reference)
        if i==0:
            valueTmp=copy.deepcopy(diarHyp)
            for y in diarHyp:
                if y['start']<(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                    if modeNoGap==False:
                        actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                        actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                    valueTmp=dropSegment(y,valueTmp)
                elif y['start']<(valueRef['start']-tolerance) and y['stop']>(valueRef['start']+tolerance):
                    actionsSegmentationBoundaryCreate.append(copy.deepcopy([y,valueRef['start']]))
                    actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([y,valueRef['start']]))
                    valueTmp=splitSegment(y,valueTmp,valueRef['start'])
                    yTmp=copy.deepcopy(y)
                    yTmp['stop']=valueRef['start']
                    if modeNoGap==False:
                        actionsSegmentationSegmentDelete.append(copy.deepcopy(yTmp))
                        actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(yTmp))
                    valueTmp=dropSegment(yTmp,valueTmp)
                elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                    # No action, all the segments in this tous les segments are dropped
                    valueTmp=dropSegment(y,valueTmp)
                else:
                    break
            # Updates diarHyp
            diarHyp=valueTmp

    # SELECTS ALL THE HYPOTHESIS SEGMENTS BETWEEN TWO REFERENCE SEGMENTS AND MAKES THEM SILENCE
        if i!=0 and diarRef[i-1]['stop']!=valueRef['start']:
            valueRefPrev=diarRef[i-1]
            valueTmp=copy.deepcopy(diarHyp)
            for y in diarHyp:
                if valueRef['start']-diarRef[i-1]['stop']<=tolerance*2:
                    # Directly deletes if the interval is smaller than tolerance*2
                    if y['start']>=(valueRefPrev['stop']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(y,valueTmp)
                    elif y['start']>=(valueRefPrev['stop']-tolerance) and y['stop']>(valueRef['start']+tolerance):
                        # Part allowing to know if we cut the segment or directly drop it
                        stopTmp=None
                        for u in range(i,len(diarRef)):
                            if y['stop']<=diarRef[u]['start']+tolerance:
                                break
                            elif y['stop']>diarRef[u]['start']+tolerance and y['stop']<=diarRef[u]['stop']+tolerance:
                                if segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                    stopTmp=diarRef[u]['start']
                                break
                            elif not segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                pass
                            else:
                                stopTmp=diarRef[u]['start']
                                break
                        if stopTmp is not None:
                            # Action here since tolerance of the valueRef segment and following ones don't crush it
                            if y['start']<(valueRef['start']-tolerance):
                                actionsSegmentationBoundaryCreate.append(copy.deepcopy([y,stopTmp]))
                                actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([y,stopTmp]))
                                valueTmp=splitSegment(y,valueTmp,stopTmp)
                                yTmp=copy.deepcopy(y)
                                yTmp['stop']=stopTmp
                                if modeNoGap==False:
                                    actionsSegmentationSegmentDelete.append(copy.deepcopy(yTmp))
                                    actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(yTmp))
                                valueTmp=dropSegment(yTmp,valueTmp)
                            break
                        else:
                            # No action since tolerance of the valueRef segment and following ones crush it
                            if modeNoGap==False:
                                actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                                actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                            valueTmp=dropSegment(y,valueTmp)               
                else:
                    if y['start']>=(valueRefPrev['stop']-tolerance) and y['start']<(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance) and y['stop']>(valueRefPrev['stop']+tolerance):
                        if modeNoGap==False:
                            actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                            actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                        valueTmp=dropSegment(y,valueTmp)
                    elif y['start']>=(valueRefPrev['stop']-tolerance) and y['start']<(valueRef['start']-tolerance) and y['stop']>(valueRef['start']+tolerance):
                        actionsSegmentationBoundaryCreate.append(copy.deepcopy([y,valueRef['start']]))
                        actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([y,valueRef['start']]))
                        valueTmp=splitSegment(y,valueTmp,valueRef['start'])
                        yTmp=copy.deepcopy(y)
                        yTmp['stop']=valueRef['start']
                        if modeNoGap==False:
                            actionsSegmentationSegmentDelete.append(copy.deepcopy(yTmp))
                            actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(yTmp))
                        valueTmp=dropSegment(yTmp,valueTmp)
                    elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(y,valueTmp)
                    elif y['start']>=(valueRef['start']-tolerance):
                        break
            # Updates diarHyp
            diarHyp=valueTmp

    # BEHAVIOR FOR A GIVEN REFERENCE SEGMENT
        # Counts the number of segment matching
        listHypRefSegment=list()
        # Whose the number in tolerance on the stop boundary
        listHypRefSegmentWithinTolerance=list()
        valueTmp=copy.deepcopy(diarHyp)
        for y in diarHyp:
            if Segment.intersection(y,valueRef) is not None:
                if tolerance==0: 
                    listHypRefSegment.append(y)
                elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                    listHypRefSegment.append(y)
                    if y['start']>=(valueRef['stop']-tolerance) and y['stop']<=(valueRef['stop']+tolerance):
                        listHypRefSegmentWithinTolerance.append(y)
        # If 0 creating
        if len(listHypRefSegment)==0 or (len(listHypRefSegment)==len(listHypRefSegmentWithinTolerance)):
            if modeNoGap == True:
                if segmentExistAccordingToTolerance(valueRef,tolerance):
                    logging.error("Cannot have absence of a segment in Transcriber mode.")
                    raise Exception("Absence of a segment.")
            if tolerance!=0:
                valueTmp2=copy.deepcopy(valueTmp)
                for u in valueTmp2:
                    if u['start']>=(valueRef['stop']-tolerance) and u['stop']<=(valueRef['stop']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(u,valueTmp)
                    elif u['start']>=(valueRef['stop']+tolerance):
                        break
            if modeNoGap == False:
                # Checks valueRef is not overtaken by tolerance
                if segmentExistAccordingToTolerance(valueRef,tolerance):
                    # Absence of the segment, so we create it
                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],valueRef['cluster'],'speakerManualNotDetected'+str(cpt+1),valueRef['start'],valueRef['stop']],['show','cluster','cluster_type','start','stop'])))                    
                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],valueRef['cluster'],'speakerManualNotDetected'+str(cpt+1),valueRef['start'],valueRef['stop']],['show','cluster','cluster_type','start','stop'])))  
                    valueTmp.append(show=showname, cluster='speakerManualNotDetected'+str(cpt+1), start=valueRef['start'], stop=valueRef['stop'])
                    cpt+=1
        # If 1 then affectation + moving boundary if need be and/or creating boundary on stop
        # If > 1 then affectation + moving boundary if need be and/or creating boundary on stop + merge
        else:
            # Checks valueRef is not overtaken by tolerance
            if not segmentExistAccordingToTolerance(valueRef,tolerance):
                for z in listHypRefSegment:
                    # Directly deletes if the interval is smaller than tolerance*2
                    if z['start']>=(valueRef['start']-tolerance) and z['stop']<=(valueRef['stop']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(z,valueTmp)
                    elif z['start']>=(valueRef['start']-tolerance) and z['stop']>(valueRef['stop']+tolerance):
                        # Part allowing to know if we cut the segment or directly drop it
                        stopTmp=None
                        for u in range(i+1,len(diarRef)):
                            if z['stop']<=diarRef[u]['start']+tolerance:
                                break
                            elif z['stop']>diarRef[u]['start']+tolerance and z['stop']<=diarRef[u]['stop']+tolerance:
                                if segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                    stopTmp=diarRef[u]['start']
                                break
                            elif not segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                pass
                            else:
                                stopTmp=diarRef[u]['start']
                                break
                        if stopTmp is not None:
                            # Action here since tolerance of the valueRef segment and following ones don't crush it
                            if z['start']<(valueRef['stop']-tolerance):
                                actionsSegmentationBoundaryCreate.append(copy.deepcopy([z,stopTmp]))
                                actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([z,stopTmp]))
                                valueTmp=splitSegment(z,valueTmp,stopTmp)
                                zTmp=copy.deepcopy(z)
                                zTmp['stop']=stopTmp
                                if modeNoGap == False:
                                    actionsSegmentationSegmentDelete.append(copy.deepcopy(zTmp))
                                    actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(zTmp))
                                valueTmp=dropSegment(zTmp,valueTmp)
                            break
                        else:
                            # No action since tolerance of the valueRef segment and following ones crush it
                            if modeNoGap == False:
                                actionsSegmentationSegmentDelete.append(copy.deepcopy(z))
                                actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(z))
                            valueTmp=dropSegment(z,valueTmp)
                # Drops the segments (left which are not in listHypRefSegment) in the tolerance margin (+ or - tolerance)
                valueTmp2=copy.deepcopy(valueTmp)
                for u in valueTmp2:
                    if u['start']>=(valueRef['stop']-tolerance) and u['stop']<=(valueRef['stop']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(u,valueTmp)
                    elif u['start']>=(valueRef['stop']+tolerance):
                        break
            else:
                # Allows to know whether we do treatments for segments with wrong boundaries
                perfectBoundary=False
                # Checks perfect boundary
                if len(listHypRefSegment)==1 and boundariesInTolerance(boundarySegment=listHypRefSegment[0],segment=valueRef,tolerance=tolerance):
                    actionsSegmentationNothing.append(copy.deepcopy(listHypRefSegment[0]))
                    actionsIncrementalSegmentationNothingTurn.append(copy.deepcopy(listHypRefSegment[0]))
                    perfectBoundary=True
                if not perfectBoundary:
                    for z in listHypRefSegment:
                        # We cut if boundary not ok to stay in the reference segment
                        if z['stop']>(valueRef['stop']+tolerance):
                            actionsSegmentationBoundaryCreate.append(copy.deepcopy([z,valueRef['stop']]))
                            actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([z,valueRef['stop']]))
                            valueTmp=splitSegment(z,valueTmp,valueRef['stop'])
                if tolerance!=0:
                    valueTmp2=copy.deepcopy(valueTmp)
                    for u in valueTmp2:
                        if u['start']>=(valueRef['stop']-tolerance) and u['stop']<=(valueRef['stop']+tolerance):
                            # No action, all the segments in this interval are dropped
                            valueTmp=dropSegment(u,valueTmp)
                        elif u['start']>=(valueRef['stop']+tolerance):
                            break
                if not perfectBoundary:
                    # Gets the new segments, modified by previous steps
                    listHypRefSegment=list()
                    # The value from where starts the segments to avoir an overlap with a previous segment which overtakes valueRef['start']
                    valueBoundaryStart=None
                    for y in valueTmp:
                        if Segment.intersection(y,valueRef) is not None:
                            if tolerance==0: 
                                listHypRefSegment.append(y)
                            elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                                listHypRefSegment.append(y)
                            elif tolerance!=0:
                                valueBoundaryStart=copy.deepcopy(y['stop'])
                    if valueBoundaryStart is None:
                        valueBoundaryStart=valueRef['start']                    
                    if modeNoGap__mergeStrat_BiggestCluster == True:
                        # Gets the cluster (it which has the most present frames)
                        dictHypRefSegmentDuration=dict()
                        for y in listHypRefSegment:
                            if y['cluster'] in dictHypRefSegmentDuration:
                                dictHypRefSegmentDuration[y['cluster']]+=y.duration()
                            else:
                                dictHypRefSegmentDuration[y['cluster']]=y.duration()
                        clusterName=max(dictHypRefSegmentDuration.keys(),key=(lambda keys: dictHypRefSegmentDuration[keys]))
                    else:
                        cls=listHypRefSegment[0]
                        for y in listHypRefSegment:
                            if cls['start']>y['start']:
                                cls=y
                        clusterName=cls['cluster']
                    # Moves the boundaries
                    # Pre-string for a good running: listHypRefSegment sorted in ascending order on start, don't overtake the value valueRef['stop'] and valueRef['start']
                    if modeNoGap == False:            
                        for idx,z in enumerate(listHypRefSegment): 
                            nearStop=valueRef['stop']
                            if idx==0:
                                boundStop=z['stop']
                            if z['stop']>=valueRef['stop']:
                                # If we reach the value of ref stop with an overlap segment
                                boundStop=valueRef['stop']
                            if boundStop!=valueRef['stop']:    
                                for r in range(idx+1,len(listHypRefSegment)):
                                    if (idx!=0 and z['stop']<=boundStop) or (z['stop']>=listHypRefSegment[r]['start'] and z['stop']<=listHypRefSegment[r]['stop']):
                                        nearStop=None
                                        break
                                    elif listHypRefSegment[r]['start']>z['stop'] and nearStop>listHypRefSegment[r]['start']:
                                        nearStop=listHypRefSegment[r]['start']
                            if nearStop is not None and boundStop!=valueRef['stop']:
                                if idx==0 and z['start']>valueRef['start'] and valueBoundaryStart!=z['start']:
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop']))) 
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop'])))
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop']))) 
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop'])))
                                    valueTmp.append(show=showname,cluster=clusterName,cluster_type=z['cluster_type'],start=valueBoundaryStart,stop=z['start'])
                                    valueTmp.append(show=showname,cluster=clusterName,cluster_type=z['cluster_type'],start=z["stop"],stop=nearStop)
                                    boundStop=nearStop
                                else:
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop'])))
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop'])))
                                    valueTmp.append(show=showname,cluster=clusterName,cluster_type=z['cluster_type'],start=z['stop'],stop=nearStop)
                                    boundStop=nearStop
                            else:
                                if idx==0 and z['start']>valueRef['start'] and valueBoundaryStart!=z['start']:
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop'])))                    
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],clusterName,z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop']))) 
                                    valueTmp.append(show=showname,cluster=clusterName,cluster_type=z['cluster_type'],start=valueBoundaryStart,stop=z['start'])
                                if boundStop<z['stop']:
                                    if z['stop']>=valueRef['stop']:
                                        boundStop=valueRef['stop']
                                    else:
                                        boundStop=z['stop']
                    # Gets the new segments, modified by the previous steps
                    listHypRefSegment=list()
                    for y in valueTmp:
                        if Segment.intersection(y,valueRef) is not None:
                            if tolerance==0: 
                                listHypRefSegment.append(y)
                            elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                                listHypRefSegment.append(y)
                    # Replaces the segments which are not in the correct cluster
                    if modeNoGap == False:
                        replaced=False
                        for y in listHypRefSegment:
                            if y['cluster']!=clusterName:
                                replaced=True
                                yTmp=copy.deepcopy(y)
                                yTmp['cluster']=clusterName
                                actionsSegmentationSegmentDelete.append(copy.deepcopy(y)) 
                                actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                                valueTmp=dropSegment(y,valueTmp)
                                actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],yTmp['cluster'],yTmp['cluster_type'],yTmp['start'],yTmp['stop']],['show','cluster','cluster_type','start','stop']))) 
                                actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],yTmp['cluster'],yTmp['cluster_type'],yTmp['start'],yTmp['stop']],['show','cluster','cluster_type','start','stop']))) 
                                valueTmp.append_seg(yTmp)  
                        if replaced:
                            valueTmp.sort()
                    # Merges among them if > 1
                    if len(listHypRefSegment)>1:
                        # Gets the new segments, modified by the previous steps
                        listTmp=list()
                        for y in valueTmp:
                            if Segment.intersection(y,valueRef) is not None:
                                if tolerance==0: 
                                    listTmp.append(y)
                                elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                                    listTmp.append(y)
                        if not (not deleteBoundarySameConsecutiveSpk and listTmp[0]['cluster']==listTmp[1]['cluster']):
                            actionsSegmentationBoundaryMerge.append(copy.deepcopy([listTmp[0],listTmp[1]]))
                            actionsIncrementalSegmentationBoundaryMergeTurn.append(copy.deepcopy([listTmp[0],listTmp[1]]))
                            if modeNoGap == True and listTmp[0]['cluster']!=listTmp[1]['cluster']:
                                listTmp[1]['cluster']=listTmp[0]['cluster']
                            newSegment,valueTmp=mergeSegment(listTmp[0],listTmp[1],valueTmp)
                        else:
                            newSegment=listTmp[1]
                        for y in range(2,len(listTmp)):
                            if modeNoGap == True:
                                if not (Segment.intersection(newSegment,listTmp[y]) is not None or newSegment["stop"]==listTmp[y]["start"] or newSegment["start"]==listTmp[y]["stop"]):
                                    logging.error("Cannot have absence of a segment in Transcriber mode.")
                                    raise Exception("Absence of a segment.")
                            if not (not deleteBoundarySameConsecutiveSpk and newSegment['cluster']==listTmp[y]['cluster']):
                                actionsSegmentationBoundaryMerge.append(copy.deepcopy([newSegment,listTmp[y]]))
                                actionsIncrementalSegmentationBoundaryMergeTurn.append(copy.deepcopy([newSegment,listTmp[y]]))
                                if modeNoGap == True and newSegment['cluster']!=listTmp[y]['cluster']:
                                    listTmp[y]['cluster']=newSegment['cluster']
                                newSegment,valueTmp=mergeSegment(newSegment,listTmp[y],valueTmp)
                            else:
                                newSegment=listTmp[y]
        # Updates diarHyp
        diarHyp=valueTmp

    # SELECTS ALL THE HYPOTHESIS SEGMENTS AFTER THE LAST REFERENCE SEGMENT (means wrong clustered since silence in the reference)
        if i==len(diarRef)-1:
            valueTmp=copy.deepcopy(diarHyp)
            for y in diarHyp:     
                if y['start']>=valueRef['stop']:
                    if modeNoGap == False:
                        actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                        actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                    valueTmp=dropSegment(y,valueTmp)
            # Updates diarHyp
            diarHyp=valueTmp
        actionsIncrementalSegmentationBoundaryCreate.append(actionsIncrementalSegmentationBoundaryCreateTurn)
        actionsIncrementalSegmentationBoundaryMerge.append(actionsIncrementalSegmentationBoundaryMergeTurn)
        if modeNoGap == False:
            actionsIncrementalSegmentationSegmentCreate.append(actionsIncrementalSegmentationSegmentCreateTurn)
            actionsIncrementalSegmentationSegmentDelete.append(actionsIncrementalSegmentationSegmentDeleteTurn)
        actionsIncrementalSegmentationNothing.append(actionsIncrementalSegmentationNothingTurn)
        actionsIncrementalSegmentationBoundaryCreateTurn=list()
        actionsIncrementalSegmentationBoundaryMergeTurn=list()
        if modeNoGap == False:
            actionsIncrementalSegmentationSegmentCreateTurn=list()
            actionsIncrementalSegmentationSegmentDeleteTurn=list()
        actionsIncrementalSegmentationNothingTurn=list()
        # Stores each diar after each human interaction
        diarIncremental[len(diarIncremental)]=(copy.deepcopy(diarHyp))
        idxIncremental[len(idxIncremental)]=(valueRef['start'],valueRef['stop'])

    # Deletes segments whose the cluster mainly matches with those present in diarFinal__clusterToDeleteAccordingToDiarRef
    for u in diarFinal__clusterToDeleteAccordingToDiarRef:
        if u in diarRef.unique("cluster"):
            diarRefTmp=diarRef.filter("cluster",'==',u)
            for t in diarHyp:
                for o in diarRefTmp:
                    if Segment.intersection(t,o) is not None:
                        match=matchingSegmentsFromSegment(t,diarRef)
                        bestMatchValue=None
                        bestMatch=None
                        if len(match)!=0:
                            for x in match:
                                if bestMatchValue is None:
                                    bestMatchValue=match[x]
                                    bestMatch=x
                                elif bestMatchValue.duration()<match[x].duration():
                                    bestMatchValue=match[x]
                                    bestMatch=x
                        diarTmp=Diar()
                        diarTmp.append_seg(t)
                        fakeTmp=Diar.intersection(diarTmp,diarRef)
                        if fakeTmp is not None:
                            fakeDuration=t.duration()-fakeTmp.duration()
                        else:
                            fakeDuration=0

                        if len(match)!=0:
                            if bestMatchValue.duration() < fakeDuration:
                                pass
                            else:
                                if bestMatch==u:
                                    diarHyp=dropSegment(t,diarHyp)                                            
     
    rtn=dict()
    rtn['idxIncremental']=idxIncremental
    rtn['diar']=dict()
    rtn['diar']['final']=diarHyp
    rtn['diar']['incremental']=diarIncremental
    rtn['action']=dict()
    rtn['action']['incremental']=actionsIncrementalSegmentationHumanCorrection
    rtn['action']['sum']=actionsSegmentationHumanCorrection

    return rtn

# Returns a dict object with an automaton which only corrects the segmentation and assignment errors
## WARNING: The diarizations in parameter have to be with no overlapped segment. Put them apart
## WARNING: The automaton follows the temporal order
## tolerance: In centiseconds
## diarFinal__clusterToDeleteAccordingToDiarRef: List of clusters to delete in the diarFinal only
## modeNoGap: Drops or not the segment actions (i.e. createSegment & deleteSegment)
## deleteBoundarySameConsecutiveSpk: Whether we delete a boundary for two consecutive segments with the same speaker
## deleteBoundaryMergeCluster: The action "delete a boundary" can merge two consecutive segments with different cluster names (it takes the name of the left/first segment)
def automatonSegmentationAssignment(diarHyp,diarRef,diarUem=None,tolerance=0,modeNoGap=False,diarFinal__clusterToDeleteAccordingToDiarRef=list(),deleteBoundarySameConsecutiveSpk=False,deleteBoundaryMergeCluster=False):
    assert isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and isinstance(modeNoGap,bool) and (diarUem is None or isinstance(diarUem,Diar)) and isinstance(tolerance,numbers.Number) and isinstance(diarFinal__clusterToDeleteAccordingToDiarRef,list) and isinstance(deleteBoundarySameConsecutiveSpk,bool) and isinstance(deleteBoundaryMergeCluster,bool)
    for u in diarFinal__clusterToDeleteAccordingToDiarRef:
        assert isinstance(u,str)

    actionsAssignmentHumanCorrection=collections.OrderedDict()
    actionsAssignmentCreate=list()
    actionsAssignmentChange=list()
    actionsAssignmentNothing=list()
    actionsAssignmentCreateBis=list()
    actionsAssignmentHumanCorrection["Create"]=actionsAssignmentCreate
    actionsAssignmentHumanCorrection["Change"]=actionsAssignmentChange
    actionsAssignmentHumanCorrection["Nothing"]=actionsAssignmentNothing
    dictionary=dict()

    actionsSegmentationHumanCorrection=collections.OrderedDict()
    actionsSegmentationBoundary=collections.OrderedDict()
    actionsSegmentationBoundaryCreate=list()
    actionsSegmentationBoundaryMerge=list()
    # Create Format: [segment, position of the new boundary] -> We have to cut the segment into two parts
    actionsSegmentationBoundary["Create"]=actionsSegmentationBoundaryCreate
    # Merge Format: [segment1, segment2] -> We have to move the segment (the two segments have to have the same attributes)
    actionsSegmentationBoundary["Merge"]=actionsSegmentationBoundaryMerge
    if modeNoGap==False:
        actionsSegmentationSegment=collections.OrderedDict()        
        actionsSegmentationSegmentCreate=list()
        actionsSegmentationSegmentDelete=list()
        # Create Format: [show,cluster,cluster_type, start, end] -> We have to create a new segment
        actionsSegmentationSegment["Create"]=actionsSegmentationSegmentCreate
        # Delete Format: [segment] -> We have to delete a segment
        actionsSegmentationSegment["Delete"]=actionsSegmentationSegmentDelete
    # Nothing Format: [segment] -> Nothing to do, correct segmentation
    actionsSegmentationNothing=list()
    actionsSegmentationHumanCorrection["Boundary"]=actionsSegmentationBoundary
    if modeNoGap==False:
        actionsSegmentationHumanCorrection["Segment"]=actionsSegmentationSegment
    actionsSegmentationHumanCorrection["Nothing"]=actionsSegmentationNothing

    actionsIncrementalAssignmentHumanCorrection=collections.OrderedDict()
    actionsIncrementalAssignmentCreate=list()
    actionsIncrementalAssignmentChange=list()
    actionsIncrementalAssignmentNothing=list()
    actionsIncrementalAssignmentHumanCorrection["Create"]=actionsIncrementalAssignmentCreate
    actionsIncrementalAssignmentHumanCorrection["Change"]=actionsIncrementalAssignmentChange
    actionsIncrementalAssignmentHumanCorrection["Nothing"]=actionsIncrementalAssignmentNothing

    actionsIncrementalSegmentationHumanCorrection=collections.OrderedDict()
    actionsIncrementalSegmentationBoundary=collections.OrderedDict()
    actionsIncrementalSegmentationBoundaryCreate=list()
    actionsIncrementalSegmentationBoundaryMerge=list()
    # Create Format: [segment, position of the new boundary] -> We have to cut the segment into two parts
    actionsIncrementalSegmentationBoundary["Create"]=actionsIncrementalSegmentationBoundaryCreate
    # Merge Format: [segment1, segment2] -> We have to move the segment (the two segments have to have the same attributes)
    actionsIncrementalSegmentationBoundary["Merge"]=actionsIncrementalSegmentationBoundaryMerge
    if modeNoGap==False:
        actionsIncrementalSegmentationSegment=collections.OrderedDict()        
        actionsIncrementalSegmentationSegmentCreate=list()
        actionsIncrementalSegmentationSegmentDelete=list()
        # Create Format: [show,cluster,cluster_type, start, end] -> We have to create a new segment
        actionsIncrementalSegmentationSegment["Create"]=actionsIncrementalSegmentationSegmentCreate
        # Delete Format: [segment] -> We have to delete a segment
        actionsIncrementalSegmentationSegment["Delete"]=actionsIncrementalSegmentationSegmentDelete
    # Nothing Format: [segment] -> Nothing to do, correct segmentation
    actionsIncrementalSegmentationNothing=list()
    actionsIncrementalSegmentationHumanCorrection["Boundary"]=actionsIncrementalSegmentationBoundary
    if modeNoGap==False:
        actionsIncrementalSegmentationHumanCorrection["Segment"]=actionsIncrementalSegmentationSegment
    actionsIncrementalSegmentationHumanCorrection["Nothing"]=actionsIncrementalSegmentationNothing

    diarIncremental=dict()

    idxIncremental=dict()

    if diarUem is not None:
        diarRef=releaseFramesAccordingToDiar(diarRef,diarUem)
        diarHyp=releaseFramesAccordingToDiar(diarHyp,diarUem)

    diarRaw=Diar()
    diarRaw.append(start=min(diarRef.unique('start')+diarHyp.unique('start')),stop=max(diarRef.unique('stop')+diarHyp.unique('stop')))
    diarRef=copy.deepcopy(diarRef)
    diarHyp=copy.deepcopy(diarHyp)
    showname=diarRef.unique('show')[0]
    diarRef.sort()
    diarHyp.sort()
    tolerance=abs(tolerance)     

    assert len(diarOverlapArea(diarRef))==0, "Error: diarRef parameter have some overlapped segments.\nReason: No overlap segment allowed.\nSolution: Please put them apart.\n"
    assert len(diarOverlapArea(diarHyp))==0, "Error: diarHyp parameter have some overlapped segments.\nReason: No overlap segment allowed.\nSolution: Please put them apart.\n"


    actionsIncrementalAssignmentCreateTurn=list()
    actionsIncrementalAssignmentChangeTurn=list()
    actionsIncrementalAssignmentNothingTurn=list()
    actionsIncrementalSegmentationBoundaryCreateTurn=list()
    actionsIncrementalSegmentationBoundaryMergeTurn=list()
    if modeNoGap==False:
        actionsIncrementalSegmentationSegmentCreateTurn=list()
        actionsIncrementalSegmentationSegmentDeleteTurn=list()
    actionsIncrementalSegmentationNothingTurn=list()

    # To avoid to create clusters with the same id
    cpt=0

    for i,valueRef in enumerate(diarRef):
    # WARNING: Each string supposes the start boundary is validate/correct (modified in the previous iteration if need be), that it doesn't overtake the reference segment (works with the tolerance as well)

    # SELECTS ALL THE HYPOTHESIS SEGMENTS BEFORE THE FIRST REFERENCE SEGMENT (means wrong clustered since silence in the reference)
        if i==0:
            valueTmp=copy.deepcopy(diarHyp)
            for y in diarHyp:
                if y['start']<(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                    if modeNoGap==False:
                        actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                        actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                    valueTmp=dropSegment(y,valueTmp)
                elif y['start']<(valueRef['start']-tolerance) and y['stop']>(valueRef['start']+tolerance):
                    actionsSegmentationBoundaryCreate.append(copy.deepcopy([y,valueRef['start']]))
                    actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([y,valueRef['start']]))
                    valueTmp=splitSegment(y,valueTmp,valueRef['start'])
                    yTmp=copy.deepcopy(y)
                    yTmp['stop']=valueRef['start']
                    if modeNoGap==False:
                        actionsSegmentationSegmentDelete.append(copy.deepcopy(yTmp))
                        actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(yTmp))
                    valueTmp=dropSegment(yTmp,valueTmp)
                elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                    # No action, all the segments in this tous les segments are dropped
                    valueTmp=dropSegment(y,valueTmp)
                else:
                    break
            # Updates diarHyp
            diarHyp=valueTmp

    # SELECTS ALL THE HYPOTHESIS SEGMENTS BETWEEN TWO REFERENCE SEGMENTS AND MAKES THEM SILENCE
        if i!=0 and diarRef[i-1]['stop']!=valueRef['start']:
            valueRefPrev=diarRef[i-1]
            valueTmp=copy.deepcopy(diarHyp)
            for y in diarHyp:
                if valueRef['start']-diarRef[i-1]['stop']<=tolerance*2:
                    # Directly deletes if the interval is smaller than tolerance*2
                    if y['start']>=(valueRefPrev['stop']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(y,valueTmp)
                    elif y['start']>=(valueRefPrev['stop']-tolerance) and y['stop']>(valueRef['start']+tolerance): 
                        # Part allowing to know if we cut the segment or directly drop it
                        stopTmp=None
                        for u in range(i,len(diarRef)):
                            if y['stop']<=diarRef[u]['start']+tolerance:
                                break
                            elif y['stop']>diarRef[u]['start']+tolerance and y['stop']<=diarRef[u]['stop']+tolerance:
                                if segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                    stopTmp=diarRef[u]['start']
                                break
                            elif not segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                pass
                            else:
                                stopTmp=diarRef[u]['start']
                                break
                        if stopTmp is not None:
                            # Action here since tolerance of the valueRef segment and following ones don't crush it
                            if y['start']<(valueRef['start']-tolerance):
                                actionsSegmentationBoundaryCreate.append(copy.deepcopy([y,stopTmp]))
                                actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([y,stopTmp]))
                                valueTmp=splitSegment(y,valueTmp,stopTmp)
                                yTmp=copy.deepcopy(y)
                                yTmp['stop']=stopTmp
                                if modeNoGap==False:
                                    actionsSegmentationSegmentDelete.append(copy.deepcopy(yTmp))
                                    actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(yTmp))
                                valueTmp=dropSegment(yTmp,valueTmp)
                            break
                        else:
                            # No action since tolerance of the valueRef segment and following ones crush it
                            if modeNoGap==False:
                                actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                                actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                            valueTmp=dropSegment(y,valueTmp)
                else:
                    if y['start']>=(valueRefPrev['stop']-tolerance) and y['start']<(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance) and y['stop']>(valueRefPrev['stop']+tolerance):
                        if modeNoGap==False:
                            actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                            actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                        valueTmp=dropSegment(y,valueTmp)
                    elif y['start']>=(valueRefPrev['stop']-tolerance) and y['start']<(valueRef['start']-tolerance) and y['stop']>(valueRef['start']+tolerance):
                        actionsSegmentationBoundaryCreate.append(copy.deepcopy([y,valueRef['start']]))
                        actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([y,valueRef['start']]))
                        valueTmp=splitSegment(y,valueTmp,valueRef['start'])
                        yTmp=copy.deepcopy(y)
                        yTmp['stop']=valueRef['start']
                        if modeNoGap==False:
                            actionsSegmentationSegmentDelete.append(copy.deepcopy(yTmp))
                            actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(yTmp))
                        valueTmp=dropSegment(yTmp,valueTmp)
                    elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance) and y['stop']<=(valueRef['start']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(y,valueTmp)
                    elif y['start']>=(valueRef['start']-tolerance):
                        break
            # Updates diarHyp
            diarHyp=valueTmp

    # BEHAVIOR FOR A GIVEN REFERENCE SEGMENT
        # Counts the number of segment matching
        listHypRefSegment=list()
        # Whose the number in tolerance on the stop boundary
        listHypRefSegmentWithinTolerance=list()
        valueTmp=copy.deepcopy(diarHyp)
        for y in diarHyp:
            if Segment.intersection(y,valueRef) is not None:
                if tolerance==0: 
                    listHypRefSegment.append(y)
                elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                    listHypRefSegment.append(y)
                    if y['start']>=(valueRef['stop']-tolerance) and y['stop']<=(valueRef['stop']+tolerance):
                        listHypRefSegmentWithinTolerance.append(y)
        # If 0 creating
        if len(listHypRefSegment)==0 or (len(listHypRefSegment)==len(listHypRefSegmentWithinTolerance)):
            if modeNoGap == True:
                if segmentExistAccordingToTolerance(valueRef,tolerance):
                    logging.error("Cannot have absence of a segment in Transcriber mode.")
                    raise Exception("Absence of a segment.")
            if tolerance!=0:
                valueTmp2=copy.deepcopy(valueTmp)
                for u in valueTmp2:
                    if u['start']>=(valueRef['stop']-tolerance) and u['stop']<=(valueRef['stop']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(u,valueTmp)
                    elif u['start']>=(valueRef['stop']+tolerance):
                        break
            if modeNoGap == False:
                # Checks valueRef is not overtaken by tolerance
                if segmentExistAccordingToTolerance(valueRef,tolerance):
                    # Absence of the segment, so we create it
                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],valueRef['cluster'],valueRef['cluster_type'],valueRef['start'],valueRef['stop']],['show','cluster','cluster_type','start','stop'])))                    
                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],valueRef['cluster'],valueRef['cluster_type'],valueRef['start'],valueRef['stop']],['show','cluster','cluster_type','start','stop'])))    
                    # Affectation part
                    if valueRef['cluster'] not in dictionary:
                        dictionary[copy.deepcopy(valueRef['cluster'])]='speakerManualNotDetected'+str(cpt+1)
                        actionsAssignmentCreateBis.append('speakerManualNotDetected'+str(cpt+1))
                        actionsAssignmentCreate.append([copy.deepcopy(valueRef['cluster']),'speakerManualNotDetected'+str(cpt+1),Segment([valueRef['show'],valueRef['cluster'],valueRef['cluster_type'],valueRef['start'],valueRef['stop']],['show','cluster','cluster_type','start','stop'])])
                        actionsIncrementalAssignmentCreateTurn.append([copy.deepcopy(valueRef['cluster']),'speakerManualNotDetected'+str(cpt+1),Segment([valueRef['show'],valueRef['cluster'],valueRef['cluster_type'],valueRef['start'],valueRef['stop']],['show','cluster','cluster_type','start','stop'])])
                        valueTmp.append(show=showname, cluster='speakerManualNotDetected'+str(cpt+1), start=valueRef['start'], stop=valueRef['stop'])
                        cpt+=1
                    else:
                        # Create with the already associated cluster
                        valueTmp.append(show=showname, cluster=dictionary[valueRef['cluster']], start=valueRef['start'], stop=valueRef['stop'])   
        # If 1 then affectation + moving boundary if need be and/or creating boundary on stop
        # If > 1 then affectation + moving boundary if need be and/or creating boundary on stop + merge
        else:
            # Checks valueRef is not overtaken by tolerance
            if not segmentExistAccordingToTolerance(valueRef,tolerance):
                for z in listHypRefSegment:
                    # Directly deletes if the interval is smaller than tolerance*2
                    if z['start']>=(valueRef['start']-tolerance) and z['stop']<=(valueRef['stop']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(z,valueTmp)
                    elif z['start']>=(valueRef['start']-tolerance) and z['stop']>(valueRef['stop']+tolerance):
                        # Part allowing to know if we cut the segment or directly drop it
                        stopTmp=None
                        for u in range(i+1,len(diarRef)):
                            if z['stop']<=diarRef[u]['start']+tolerance:
                                break
                            elif z['stop']>diarRef[u]['start']+tolerance and z['stop']<=diarRef[u]['stop']+tolerance:
                                if segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                    stopTmp=diarRef[u]['start']
                                break
                            elif not segmentExistAccordingToTolerance(diarRef[u],tolerance):
                                pass
                            else:
                                stopTmp=diarRef[u]['start']
                                break
                        if stopTmp is not None:
                            # Action here since tolerance of the valueRef segment and following ones don't crush it
                            if z['start']<(valueRef['stop']-tolerance):
                                actionsSegmentationBoundaryCreate.append(copy.deepcopy([z,stopTmp]))
                                actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([z,stopTmp]))
                                valueTmp=splitSegment(z,valueTmp,stopTmp)
                                zTmp=copy.deepcopy(z)
                                zTmp['stop']=stopTmp
                                if modeNoGap == False:
                                    actionsSegmentationSegmentDelete.append(copy.deepcopy(zTmp))
                                    actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(zTmp))
                                valueTmp=dropSegment(zTmp,valueTmp)
                            break
                        else:
                            # No action since tolerance of the valueRef segment and following ones crush it
                            if modeNoGap == False:
                                actionsSegmentationSegmentDelete.append(copy.deepcopy(z))
                                actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(z))
                            valueTmp=dropSegment(z,valueTmp)
                # Drops the segments (left which are not in listHypRefSegment) in the tolerance margin (+ or - tolerance)
                valueTmp2=copy.deepcopy(valueTmp)
                for u in valueTmp2:
                    if u['start']>=(valueRef['stop']-tolerance) and u['stop']<=(valueRef['stop']+tolerance):
                        # No action, all the segments in this interval are dropped
                        valueTmp=dropSegment(u,valueTmp)
                    elif u['start']>=(valueRef['stop']+tolerance):
                        break
            else:
                # Allows to know whether we do treatments for segments with wrong boundaries
                perfectBoundary=False
                # Checks perfect boundary
                if len(listHypRefSegment)==1 and boundariesInTolerance(boundarySegment=listHypRefSegment[0],segment=valueRef,tolerance=tolerance):
                    actionsSegmentationNothing.append(copy.deepcopy(listHypRefSegment[0]))
                    actionsIncrementalSegmentationNothingTurn.append(copy.deepcopy(listHypRefSegment[0]))
                    perfectBoundary=True
                if not perfectBoundary:
                    for z in listHypRefSegment:
                        # We cut if boundary not ok to stay in the reference segment
                        if z['stop']>(valueRef['stop']+tolerance):
                            actionsSegmentationBoundaryCreate.append(copy.deepcopy([z,valueRef['stop']]))
                            actionsIncrementalSegmentationBoundaryCreateTurn.append(copy.deepcopy([z,valueRef['stop']]))
                            valueTmp=splitSegment(z,valueTmp,valueRef['stop'])
                if tolerance!=0:
                    valueTmp2=copy.deepcopy(valueTmp)
                    for u in valueTmp2:
                        if u['start']>=(valueRef['stop']-tolerance) and u['stop']<=(valueRef['stop']+tolerance):
                            # No action, all the segments in this interval are dropped
                            valueTmp=dropSegment(u,valueTmp)
                        elif u['start']>=(valueRef['stop']+tolerance):
                            break
                # Gets the new segments, modified by previous steps
                listHypRefSegment=list()
                for y in valueTmp:
                    if Segment.intersection(y,valueRef) is not None:
                        if tolerance==0: 
                            listHypRefSegment.append(y)
                        elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                            listHypRefSegment.append(y)
                for idx,z in enumerate(listHypRefSegment):
                    # Affectation part
                    applyChange=False
                    if valueRef['cluster'] not in dictionary:
                        if z['cluster'] in actionsAssignmentCreateBis:
                            dictionary[copy.deepcopy(valueRef['cluster'])]='speakerManual'+str(cpt+1)
                            actionsAssignmentCreateBis.append('speakerManual'+str(cpt+1))
                            actionsAssignmentCreate.append([copy.deepcopy(valueRef['cluster']),'speakerManual'+str(cpt+1),copy.deepcopy(z)])
                            actionsIncrementalAssignmentCreateTurn.append([copy.deepcopy(valueRef['cluster']),'speakerManual'+str(cpt+1),copy.deepcopy(z)])
                            applyChange=True
                            cpt+=1
                        else:
                            dictionary[copy.deepcopy(valueRef['cluster'])]=copy.deepcopy(z['cluster'])
                            actionsAssignmentCreateBis.append(copy.deepcopy(z['cluster']))
                            actionsAssignmentCreate.append(copy.deepcopy([valueRef['cluster'],z['cluster'],copy.deepcopy(z)]))
                            actionsIncrementalAssignmentCreateTurn.append(copy.deepcopy([valueRef['cluster'],z['cluster'],copy.deepcopy(z)]))
                    else:
                        if z['cluster'] == dictionary[valueRef['cluster']]:
                            actionsAssignmentNothing.append(copy.deepcopy(z))
                            actionsIncrementalAssignmentNothingTurn.append(copy.deepcopy(z))
                        else:
                            actionsAssignmentChange.append(copy.deepcopy([dictionary[valueRef['cluster']],z]))
                            actionsIncrementalAssignmentChangeTurn.append(copy.deepcopy([dictionary[valueRef['cluster']],z]))
                            applyChange=True
                    if applyChange:
                        # Updates the diar for the merges afterward
                        segmentTmp=copy.deepcopy(z)
                        segmentTmp['cluster']=dictionary[valueRef['cluster']]
                        valueTmp=dropSegment(z,valueTmp)
                        valueTmp.append_seg(segmentTmp)
                        valueTmp.sort()
                    if deleteBoundaryMergeCluster:
                        break
                if not perfectBoundary:
                    # Gets the new segments, modified by the previous steps
                    listHypRefSegment=list()
                    # The value from where starts the segments to avoir an overlap with a previous segment which overtakes valueRef['start']
                    valueBoundaryStart=None
                    for y in valueTmp:
                        if Segment.intersection(y,valueRef) is not None:
                            if tolerance==0: 
                                listHypRefSegment.append(y)
                            elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                                listHypRefSegment.append(y)
                            elif tolerance!=0:
                                valueBoundaryStart=copy.deepcopy(y['stop'])
                    if valueBoundaryStart is None:
                        valueBoundaryStart=valueRef['start']
                    if modeNoGap == False:        
                        for idx,z in enumerate(listHypRefSegment): 
                            # Moves the boundaries
                            # Pre-string for a good running: listHypRefSegment sorted in ascending order on start, don't overtake the value valueRef['stop'] and valueRef['start']
                            nearStop=valueRef['stop']
                            if idx==0:
                                boundStop=z['stop']
                            if z['stop']>=valueRef['stop']:
                                # If we reach the value of ref stop with an overlap segment
                                boundStop=valueRef['stop']                                
                            if boundStop!=valueRef['stop']:    
                                for r in range(idx+1,len(listHypRefSegment)):
                                    if (idx!=0 and z['stop']<=boundStop) or (z['stop']>=listHypRefSegment[r]['start'] and z['stop']<=listHypRefSegment[r]['stop']):
                                        nearStop=None
                                        break
                                    elif listHypRefSegment[r]['start']>z['stop'] and nearStop>listHypRefSegment[r]['start']:
                                        nearStop=listHypRefSegment[r]['start']
                            if nearStop is not None and boundStop!=valueRef['stop']:
                                if idx==0 and z['start']>valueRef['start'] and valueBoundaryStart!=z['start']:
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop']))) 
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop']))) 
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop']))) 
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop']))) 
                                    valueTmp.append(show=showname,cluster=z['cluster'],cluster_type=z['cluster_type'],start=valueBoundaryStart,stop=z['start'])
                                    valueTmp.append(show=showname,cluster=z['cluster'],cluster_type=z['cluster_type'],start=z["stop"],stop=nearStop)
                                    boundStop=nearStop
                                else:
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop']))) 
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],z['stop'],nearStop],['show','cluster','cluster_type','start','stop']))) 
                                    valueTmp.append(show=showname,cluster=z['cluster'],cluster_type=z['cluster_type'],start=z['stop'],stop=nearStop)
                                    boundStop=nearStop
                            else:
                                if idx==0 and z['start']>valueRef['start'] and valueBoundaryStart!=z['start']:
                                    actionsSegmentationSegmentCreate.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop'])))                    
                                    actionsIncrementalSegmentationSegmentCreateTurn.append(copy.deepcopy(Segment([valueRef['show'],z['cluster'],z['cluster_type'],valueBoundaryStart,z['start']],['show','cluster','cluster_type','start','stop'])))  
                                    valueTmp.append(show=showname,cluster=z['cluster'],cluster_type=z['cluster_type'],start=valueBoundaryStart,stop=z['start'])
                                if boundStop<z['stop']:
                                    if z['stop']>=valueRef['stop']:
                                        boundStop=valueRef['stop']
                                    else:
                                        boundStop=z['stop']
                    # Gets the new segments, modified by the previous steps
                    listHypRefSegment=list()
                    for y in valueTmp:
                        if Segment.intersection(y,valueRef) is not None:
                            if tolerance==0: 
                                listHypRefSegment.append(y)
                            elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                                listHypRefSegment.append(y)
                    # Merges among them if > 1
                    if len(listHypRefSegment)>1:
                        # Gets the new segments, modified by the previous steps
                        listTmp=list()
                        for y in valueTmp:
                            if Segment.intersection(y,valueRef) is not None:
                                if tolerance==0: 
                                    listTmp.append(y)
                                elif tolerance!=0 and y['start']>=(valueRef['start']-tolerance):
                                    listTmp.append(y)
                        if not (not deleteBoundarySameConsecutiveSpk and listTmp[0]['cluster']==listTmp[1]['cluster']):
                            actionsSegmentationBoundaryMerge.append(copy.deepcopy([listTmp[0],listTmp[1]]))
                            actionsIncrementalSegmentationBoundaryMergeTurn.append(copy.deepcopy([listTmp[0],listTmp[1]]))
                            if modeNoGap == True and listTmp[0]['cluster']!=listTmp[1]['cluster']:
                                listTmp[1]['cluster']=listTmp[0]['cluster']
                            newSegment,valueTmp=mergeSegment(listTmp[0],listTmp[1],valueTmp)
                        else:
                            newSegment=listTmp[1]
                        for y in range(2,len(listTmp)):
                            if modeNoGap == True:
                                if not (Segment.intersection(newSegment,listTmp[y]) is not None or newSegment["stop"]==listTmp[y]["start"] or newSegment["start"]==listTmp[y]["stop"]):
                                    logging.error("Cannot have absence of a segment in Transcriber mode.")
                                    raise Exception("Absence of a segment.")
                            if not (not deleteBoundarySameConsecutiveSpk and newSegment['cluster']==listTmp[y]['cluster']):
                                actionsSegmentationBoundaryMerge.append(copy.deepcopy([newSegment,listTmp[y]]))
                                actionsIncrementalSegmentationBoundaryMergeTurn.append(copy.deepcopy([newSegment,listTmp[y]]))
                                if modeNoGap == True and newSegment['cluster']!=listTmp[y]['cluster']:
                                    valueTmp=dropSegment(listTmp[y],valueTmp)
                                    listTmp[y]['cluster']=newSegment['cluster']
                                    valueTmp.append_seg(listTmp[y])
                                    valueTmp.sort()
                                newSegment,valueTmp=mergeSegment(newSegment,listTmp[y],valueTmp)
                            else:
                                newSegment=listTmp[y]                    
        # Updates diarHyp
        diarHyp=valueTmp

    # SELECTS ALL THE HYPOTHESIS SEGMENTS AFTER THE LAST REFERENCE SEGMENT (means wrong clustered since silence in the reference)
        if i==len(diarRef)-1:
            valueTmp=copy.deepcopy(diarHyp)
            for y in diarHyp:     
                if y['start']>=valueRef['stop']:
                    if modeNoGap == False:
                        actionsSegmentationSegmentDelete.append(copy.deepcopy(y))
                        actionsIncrementalSegmentationSegmentDeleteTurn.append(copy.deepcopy(y))
                    valueTmp=dropSegment(y,valueTmp)
            # Updates diarHyp
            diarHyp=valueTmp
        actionsIncrementalAssignmentCreate.append(actionsIncrementalAssignmentCreateTurn)
        actionsIncrementalAssignmentChange.append(actionsIncrementalAssignmentChangeTurn)
        actionsIncrementalAssignmentNothing.append(actionsIncrementalAssignmentNothingTurn)
        actionsIncrementalSegmentationBoundaryCreate.append(actionsIncrementalSegmentationBoundaryCreateTurn)
        actionsIncrementalSegmentationBoundaryMerge.append(actionsIncrementalSegmentationBoundaryMergeTurn)
        if modeNoGap == False:
            actionsIncrementalSegmentationSegmentCreate.append(actionsIncrementalSegmentationSegmentCreateTurn)
            actionsIncrementalSegmentationSegmentDelete.append(actionsIncrementalSegmentationSegmentDeleteTurn)
        actionsIncrementalSegmentationNothing.append(actionsIncrementalSegmentationNothingTurn)
        actionsIncrementalAssignmentCreateTurn=list()
        actionsIncrementalAssignmentChangeTurn=list()
        actionsIncrementalAssignmentNothingTurn=list()
        actionsIncrementalSegmentationBoundaryCreateTurn=list()
        actionsIncrementalSegmentationBoundaryMergeTurn=list()
        if modeNoGap == False:
            actionsIncrementalSegmentationSegmentCreateTurn=list()
            actionsIncrementalSegmentationSegmentDeleteTurn=list()
        actionsIncrementalSegmentationNothingTurn=list()
        # Stores each diar after each human interaction
        diarIncremental[len(diarIncremental)]=(copy.deepcopy(diarHyp))
        idxIncremental[len(idxIncremental)]=(valueRef['start'],valueRef['stop'])

    # Deletes segments whose the cluster mainly matches with those present in diarFinal__clusterToDeleteAccordingToDiarRef
    for u in diarFinal__clusterToDeleteAccordingToDiarRef:
        if u in dictionary:
            diarHyp=dropCluster(dictionary[u],diarHyp)
    
    rtn=dict()
    rtn['idxIncremental']=idxIncremental
    rtn['diar']=dict()
    rtn['diar']['final']=diarHyp
    rtn['diar']['incremental']=diarIncremental
    rtn['action']=dict()
    rtn['action']['incremental']=dict()
    rtn['action']['incremental']['assignment']=actionsIncrementalSegmentationHumanCorrection
    rtn['action']['incremental']['segmentation']=actionsIncrementalAssignmentHumanCorrection
    rtn['action']['sum']=dict()
    rtn['action']['sum']['assignment']=actionsAssignmentHumanCorrection
    rtn['action']['sum']['segmentation']=actionsSegmentationHumanCorrection

    return rtn

# Returns a border structure from an hypothesis diar object and a reference diar object according to the uem diar object
## A border is a list which matches the labels from the hypothesis and the reference
## tolerance: In centiseconds
def border(diarHyp,diarRef,diarUem=None,tolerance=0,nonSpeechNameRef="NONSPEECHREF",nonSpeechNameHyp="NONSPEECHHYP",toleranceNameRef="TOLERANCE",toleranceNameHyp="TOLERANCE"):
    assert isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and (diarUem is None or isinstance(diarUem,Diar)) and isinstance(nonSpeechNameRef,str) and isinstance(nonSpeechNameHyp,str) and isinstance(tolerance,numbers.Number) and tolerance>=0
    if diarUem is not None:
        diarUem=compressDiar(diarUem)
        uem_start= min(diarUem.unique('start'))
        uem_stop = max(diarUem.unique('stop'))
        diarRef=releaseFramesAccordingToDiar(diarRef,diarUem)
        diarHyp=releaseFramesAccordingToDiar(diarHyp,diarUem)
    else:
        uem_start= min(diarRef.unique('start')+diarHyp.unique('start'))
        uem_stop = max(diarRef.unique('stop')+diarHyp.unique('stop'))
        diarUem=Diar()
        diarUem.append(start=uem_start,stop=uem_stop)

    # Overtakes the clusters for the tolerance if need be
    if tolerance >0:
        oldDiarRef=copy.deepcopy(diarRef)
        diarHyp=releaseFramesAccordingToDiarWithToleranceBoundaries(diarHyp,oldDiarRef,tolerance)
        diarRef=releaseFramesAccordingToDiarWithToleranceBoundaries(diarRef,oldDiarRef,tolerance)
        for idx,i in enumerate(oldDiarRef):
            if idx==0:
                diarHyp.append(start=i['start'],stop=i['stop']+tolerance,cluster=toleranceNameHyp)
                diarRef.append(start=i['start'],stop=i['stop']+tolerance,cluster=toleranceNameRef)
            elif idx==len(oldDiarRef-1):
                diarHyp.append(start=i['start']-tolerance,stop=i['stop'],cluster=toleranceNameHyp)
                diarRef.append(start=i['start']-tolerance,stop=i['stop'],cluster=toleranceNameRef)
            else:
                diarHyp.append(start=i['start']-tolerance,stop=i['stop']+tolerance,cluster=toleranceNameHyp)
                diarRef.append(start=i['start']-tolerance,stop=i['stop']+tolerance,cluster=toleranceNameRef)

    # Fills the wholes to be comparable
    ref_diar=fillDiar(nonSpeechNameRef,diarUem,diarRef)
    hyp_diar=fillDiar(nonSpeechNameHyp,diarUem,diarHyp)
    
    # Creates vectors
    ref_vect = vector(ref_diar,start=uem_start,stop=uem_stop)
    hyp_vect = vector(hyp_diar,start=uem_start,stop=uem_stop)

    # Creates the Border
    border = collections.OrderedDict()

    # Selects the components of the Border by avoiding repetitions
    for j in diarUem:
        border[j['start']] = [j['start'], ref_vect[j['start']], hyp_vect[j['start']], 'start', -1]
        for i in sorted(range(j['start']+1,j['stop'])):
            if ref_vect[i-1] != ref_vect[i]:
                k = 'ref'
                if hyp_vect[i-1] != hyp_vect[i]:
                    k = 'refhyp'
                border[i] = [i, ref_vect[i], hyp_vect[i], k , -1]
            elif hyp_vect[i-1] != hyp_vect[i]:
                border[i] = [i, ref_vect[i], hyp_vect[i], 'hyp', -1]

    # Converts the Border into list
    border_list = list()
    for idx in border:
        border_list.append(border[idx])

    # Updates the "stop" of each Border component
    for j in diarUem:
        listTmp=list()
        for i in range(0, len(border_list)):
            if border_list[i][0]>=j['start'] and border_list[i][0]<=j['stop']:
                listTmp.append(i)
            elif border_list[i][0]>j['stop']:
                break
        for idx,i in enumerate(listTmp):
            if idx!=0:
                l = border_list[i][0] - border_list[i-1][0]
                border_list[i-1][4] = l
            if idx==len(listTmp)-1:
                border_list[i][4] = j['stop'] - border_list[i][0]

    return border_list

# Returns the HCIQ measure for a border structure
## tolerance: In centiseconds
## cb (createBoundary), db (deleteBoundary), cn (createName), sn (selectName), v (validate) are weights
def borderHCIQ(diarHyp,diarRef,diarUem=None,cb=1,db=1,cn=1,sn=1,v=0,tolerance=0,nameAlreadySet=set(),nonSpeechNameRef="NONSPEECHREF",nonSpeechNameHyp="NONSPEECHHYP",toleranceNameRef="TOLERANCE",toleranceNameHyp="TOLERANCE"):
    assert isinstance(cb,numbers.Number) and isinstance(db,numbers.Number) and isinstance(cb,numbers.Number) and isinstance(cn,numbers.Number) and isinstance(v,numbers.Number) and isinstance(sn,numbers.Number)
    assert cb>=0 and db>=0 and cn>=0 and sn>=0 and v>=0
    get=borderHumanCorrectionsToDo(diarHyp,diarRef,diarUem,False,nameAlreadySet,tolerance,nonSpeechNameRef,nonSpeechNameHyp,toleranceNameRef,toleranceNameHyp)
    HCIQ=(get["create_boundary"]*cb)+(get["delete_boundary"]*db)+(get["create_name"]*cn)+(get["select_name"]*sn)+(get["validate"]*v)
    return HCIQ

# Returns a dict object with the human corrections to do on the diarHyp parameter according to the diarRef and diarUem parameters for a border structure
def borderHumanCorrectionsToDo(diarHyp,diarRef,diarUem=None,verbose=False,nameAlreadySet=set(),tolerance=0,nonSpeechNameRef="NONSPEECHREF",nonSpeechNameHyp="NONSPEECHHYP",toleranceNameRef="TOLERANCE",toleranceNameHyp="TOLERANCE"):
    assert isinstance(verbose,bool) and isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and (diarUem is None or isinstance(diarUem,Diar)) and isinstance(nameAlreadySet,set)
    if diarUem is not None:
        diarUem=compressDiar(diarUem)
    else:
        uem_start= min(diarRef.unique('start')+diarHyp.unique('start'))
        uem_stop = max(diarRef.unique('stop')+diarHyp.unique('stop'))
        diarUem=Diar()
        diarUem.append(start=uem_start,stop=uem_stop)

    # Creates the Border
    border_list=border(diarHyp,diarRef,diarUem,tolerance,nonSpeechNameRef,nonSpeechNameHyp,toleranceNameRef,toleranceNameHyp)

    # Initializes action counters
    output=dict()
    output["validate"]=0
    output["create_boundary"]=0
    output["delete_boundary"]=0
    output["create_name"]=0
    output["select_name"]=0
    output["name_set"]=copy.deepcopy(nameAlreadySet)

    for z in diarUem:
        listTmp=list()
        for i in range(0, len(border_list)):
            if border_list[i][0]>=z['start'] and border_list[i][0]<=z['stop']:
                listTmp.append(i)
            elif border_list[i][0]>z['stop']:
                break
        if verbose:
            logging.info('='*10)
        for idx,i in enumerate(listTmp):
            if idx==0:
                prev_idx, prev_ref, prev_hyp, prev_border, prev_l = border_list[i]

                # Checks the first identical ref and hyp 
                if prev_hyp != prev_ref:
                    if prev_ref in output["name_set"]:
                        # Selects name if existing
                        output["select_name"] += 1
                        border_list[i][2] = prev_ref
                    else:
                        # Creates name if existing
                        output["create_name"] += 1
                        if prev_hyp not in output["name_set"]:
                            borderRenameAll(border_list, i, prev_ref)
                        else:
                            border_list[i][2] = prev_ref
                        output["name_set"].add(prev_ref)
                    
                    prev_idx, prev_ref, prev_hyp, prev_border, prev_l = border_list[i]
                    if verbose: 
                        logging.info(prev_border, prev_idx, prev_l, 'ref=', prev_ref, 'hyp=', hyp, 'new hyp=', prev_hyp, 'cmp=', hyp == prev_ref , 'v=', output["validate"], 'cb=', output["create_boundary"], 'db=', output["delete_boundary"], 'cn=', output["create_name"], 'sn=', output["select_name"])
                else:
                    # Validates if existing
                    output["validate"] += 1
                    
                    if verbose:   
                        logging.info(prev_idx, ' ref=', prev_ref, ' hyp=', prev_hyp, ' cmp=', prev_ref==prev_hyp , 'v=', output["validate"], 'cb=', output["create_boundary"], 'db=', output["delete_boundary"], 'cn=', output["create_name"], 'sn=', output["select_name"])
                if verbose:
                    logging.info('-'*10)
            else:
                idx, ref, hyp, border, l = border_list[i]
                if verbose:   
                    logging.info(prev_idx, 'vs', idx,';', prev_ref, 'vs', ref, ';', prev_hyp,'vs', hyp)
                if hyp == ref:
                    # ref A A
                    # hyp A A
                    output["validate"] += 1
                    if verbose: 
                        logging.info('validate ref : A A / hyp A A')
                else :
                    if prev_ref == ref:
                        # ref : A A
                        # hyp : A B
                        # B to A
                        borderRenameFollowing(border_list,i,ref)
                        output["delete_boundary"] += 1
                        output["name_set"].add(ref)
                        if verbose:   
                            logging.info('delete ref : A A / hyp A B')
                    else:
                        if prev_hyp == hyp:
                            # ref : A B
                            # hyp : A A
                            # create boundary and create/select name
                            output["create_boundary"] += 1
                            if verbose:   
                                logging.info('create_boundary ref : A B / hyp A A')

                        # ref : A B
                        # hyp : A C/A
                        # create/select name
                        if ref in output["name_set"]:
                            output["select_name"] += 1
                            border_list[i][2] = ref
                            if verbose:   
                                logging.info('select_name ref : A B / hyp A A/C')
                        else:
                            output["create_name"] += 1
                            if hyp not in output["name_set"]:
                                borderRenameAll(border_list, i, ref)
                            else:
                                border_list[i][2]=ref
                            output["name_set"].add(ref)
                            if verbose:   
                                logging.info('create_name ref : A B / hyp A A/C', output["name_set"])

                prev_idx, prev_ref, prev_hyp, prev_border, prev_l = border_list[i]
                if verbose: 
                    logging.info(prev_border, prev_idx, prev_l, 'ref=', prev_ref, 'hyp=', hyp, 'new hyp=', prev_hyp, 'cmp=', hyp == prev_ref , 'v=', output["validate"], 'cb=', output["create_boundary"], 'db=', output["delete_boundary"], 'cn=', output["create_name"], 'sn=', output["select_name"], '\n')
    return output

# Renames all the names identical to the one at the "idx" position by "name" for a border structure
def borderRenameAll(border, idx, name):
    assert isinstance(border,list) and isinstance(idx,numbers.Number) and isinstance(name,str)
    old = border[idx][2]
    for j in range(idx, len(border)):
        if border[j][2] == old:
            border[j][2] = name

# Renames only if the following borders at the "idx" position have the same name as the previous one for a border structure
def borderRenameFollowing(border,idx,name):
    assert isinstance(border,list) and isinstance(idx,numbers.Number) and isinstance(name,str)
    oldN=border[idx][2]
    oldV=border[idx][0]+border[idx][4]
    border[idx][2]=name
    for j in range(idx+1, len(border)):
        if border[j][0]==oldV and border[j][2] == oldN:
            border[j][2] = name
            oldV=border[j][0]+border[j][4]
        else:
            break

# Checks if the boundarySegment parameter is in the tolerance of the segment parameter
## tolerance: In centiseconds
def boundariesInTolerance(boundarySegment,segment,tolerance):
    assert isinstance(segment,Segment) and isinstance(tolerance,numbers.Number) and isinstance(boundarySegment,Segment)
    if (boundarySegment['start']<=segment['start']+tolerance and boundarySegment['start']>=segment['start']-tolerance) and (boundarySegment['stop']<=segment['stop']+tolerance and boundarySegment['stop']>=segment['stop']-tolerance):
        return True
    return False

# Returns a dictionary object which represents the reference boundaries linked to the hypothesis boundaries
## WARNING: The boundary matching rests on the nearest distance. In any case, it doesn't take into consideration the labels
## tolerance: In centiseconds
def boundHypToChange(diarHyp,diarRef,diarUem=None,verbose=False,tolerance=25):
    assert isinstance(verbose,bool) and isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and ((isinstance(diarUem,Diar) and len(diarOverlapArea(diarUem))==0) or diarUem is None) and len(diarOverlapArea(diarRef))==0 and len(diarOverlapArea(diarHyp))==0  
    diarHyp=copy.deepcopy(diarHyp)
    diarRef=copy.deepcopy(diarRef)
    if diarUem is not None:
        diarUem=compressDiar(diarUem)
        diarRef=releaseFramesAccordingToDiar(diarRef,diarUem)
        diarHyp=releaseFramesAccordingToDiar(diarHyp,diarUem)
    boundHyp=set(diarHyp.unique("stop")).union(set(diarHyp.unique("start")))
    boundRef=set(diarRef.unique("stop")).union(set(diarRef.unique("start")))
    boundHyp=list(boundHyp)
    boundRef=list(boundRef)
    boundHyp.sort()
    boundRef.sort()

    affected=dict()
    for i in boundRef:
        affected[i]=None

    matrix=np.full((len(boundRef), len(boundHyp)), tolerance+1000)
    for idxI,i in enumerate(sorted(boundRef)):
        for idxJ,j in enumerate(sorted(boundHyp)):
            val=abs(i-j)
            if abs(val)<=tolerance:
                matrix[idxI,idxJ]=val
            else:
                matrix[idxI,idxJ]=tolerance+1000
    if diarUem is not None:
        matrixTmp=np.full((len(boundRef), len(boundHyp)), tolerance+1000)
        for y in diarUem:
            for idxI,i in enumerate(sorted(boundRef)):
                for idxJ,j in enumerate(sorted(boundHyp)):
                    if i>=y['start'] and i<=y['stop'] and j>=y['start'] and j<=y['stop']:
                        matrixTmp[idxI,idxJ]=matrix[idxI,idxJ]
        matrix=matrixTmp
           
    if verbose:
        print(boundHyp)
        print(boundRef)

    while matrix.min() <= tolerance:
        jPos=None
        iPos=None
        if verbose:
            print("---")
            print(affected)
            print(matrix)
            print(matrix.min())
        for i in range(0,len(matrix)):
            for j in range(0,len(matrix[i])):
                if matrix[i,j]==matrix.min():
                    jPos=j
                    iPos=i
                    break
            if jPos is not None:
                break
        if verbose:
            print("LIGNE:",iPos)
            print("COLONNE:",jPos)
        boundHypPosVal=boundHyp[jPos]
        boundRefPosVal=boundRef[iPos]
        if verbose:
            print("HYP",boundHypPosVal)
            print("REF",boundRefPosVal)
        cross=False
        for y in affected:
            if affected[y] is not None and y!=boundRefPosVal:
                if not ((y<boundRefPosVal and affected[y]<boundHypPosVal) or (y>boundRefPosVal and affected[y]>boundHypPosVal)):
                    cross=True
        if not cross:
            affected[boundRefPosVal]=boundHypPosVal
            for y in range(0,len(boundHyp)):
                matrix[iPos,y]=1000+tolerance
        matrix[iPos,jPos]=1000+tolerance
    return affected

# Checks the structure of a diar object
def certifyDiar(diar):
    if not isinstance(diar,Diar):
        return False
    for i in diar:
        if not isinstance(i,Segment):
            print(i.__class__)
            return False
    return True

# Checks if two diar objects are identical
def compareDiar(diar1,diar2):
    assert isinstance(diar1,Diar) and isinstance(diar2,Diar)
    diar1=copy.deepcopy(diar1)
    diar2=copy.deepcopy(diar2)
    diar1.sort()
    diar2.sort()
    if len(diar1)!=len(diar2):
        return False
    for j,val in enumerate(diar1):
        if not (diar1[j]['show']==diar2[j]['show'] and diar1[j]['cluster']==diar2[j]['cluster'] and diar1[j]['cluster_type']==diar2[j]['cluster_type'] and diar1[j]['start']==diar2[j]['start'] and diar1[j]['stop']==diar2[j]['stop']):
            return False
    return True

# Returns a compressed diar object
def compressDiar(diar):
    assert isinstance(diar,Diar)
    diarOut=copy.deepcopy(diar)
    for i in diarOut:
        i["show"]="show"
        i["cluster"]="cluster"
    diarOut.pack()
    return diarOut

# Returns the occurence of a given segment
def countOccurenceSegment(segment,diar):
    assert isinstance(segment,Segment) and isinstance(diar,Diar)
    cpt=0
    for i in diar:
        if i['show']==segment['show'] and i['cluster']==segment['cluster'] and i['cluster_type']==segment['cluster_type'] and i['start']==segment['start'] and i['stop']==segment['stop']:
            cpt+=1
    return cpt

# Returns a diar object by cutting with a rolling mean in low energy area in order to only have segments with a given max size 
## winSize, maxSegSize, securityMarginSize: In centiseconds
## c0 of cepstrum have to be the energy
def cutBigSegmentLowEnergy(diar,cepstrum,maxSegSize,securityMarginSize,winSize=100):
    assert isinstance(diar,Diar) and isinstance(maxSegSize,numbers.Number) and isinstance(securityMarginSize,numbers.Number) and isinstance(winSize,numbers.Number)
    assert (securityMarginSize*2)<maxSegSize and (winSize/2)<=securityMarginSize
    flag=False
    diar=copy.deepcopy(diar)
    outputDiar=copy.deepcopy(diar)
    # Computes rolling mean and std in the window of size win, gets numpy array
    # Mean and std have NAN at the beginning and the end of the output array
    df = pd.DataFrame(cepstrum)
    r = df.rolling(window=int(winSize/2), center=False)
    mean = r.mean().values
    for i in diar:
        if i.duration()>maxSegSize:
            value=i['start']+securityMarginSize
            for y in range(i['start']+securityMarginSize,i['stop']-securityMarginSize,1):
                if mean[y][0]<mean[value][0]:
                    value=y
            outputDiar=splitSegment(i,outputDiar,value)
            if (value-i['start'])>maxSegSize or (i['stop']-value)>maxSegSize:
                flag=True
    if flag:
        return cutBigSegmentLowEnergy(outputDiar,cepstrum,maxSegSize,securityMarginSize,winSize)
    else:
        return outputDiar

# Returns a diar object with the overlap areas
def diarOverlapArea(diar):
    assert isinstance(diar,Diar)
    out_diar_tmp=copy.deepcopy(diar)
    idx = out_diar_tmp.features_by_cluster()
    clusters = out_diar_tmp.unique('cluster')
    show = out_diar_tmp.unique('show')

    # Last frame
    l = max(out_diar_tmp.unique('stop'))

    # Counts the speaker number by frame
    s = np.zeros(l)
    for cluster in clusters:
        s[idx[cluster]] += 1

    # Vector of indexes counting more than one speaker
    npTmp=np.squeeze(np.asarray(np.argwhere(s > 1)))
    if npTmp.size<=1:
        notUnique=set()
        if npTmp.size==1:
            notUnique.add(int(npTmp))
    else:
        notUnique = set(npTmp)

    out_diar = Diar()
    for cluster in clusters:
        for i in list(set(idx[cluster]) & notUnique):
            out_diar.append(show=show[0], cluster="OVERLAP", start=i, stop=i+1)
    out_diar.pack()
    for i in out_diar:
        listTmp=set()
        for j in out_diar_tmp:
            if Segment.intersection(i,j) is not None:
                listTmp.add(j["cluster"])
        listTmp=list(listTmp)
        listTmp.sort()
        cluster=listTmp[0]
        for j in range(1,len(listTmp)):
            cluster+=(" / "+listTmp[j])
        i["cluster"]=cluster
    return out_diar

# Returns a diar object without the segments having the cluster name put in parameter
def dropCluster(cluster,diar):
    assert isinstance(cluster,str) and isinstance(diar,Diar)
    out_diar=Diar()
    for i in diar:
        if i['cluster']==cluster:
            pass
        else:
            out_diar.append_seg(i)
    return copy.deepcopy(out_diar)

# Returns a diar object without the segment put in parameter
def dropSegment(segment,diar):
    assert isinstance(segment,Segment) and isinstance(diar,Diar) and countOccurenceSegment(segment,diar)==1
    out_diar=Diar()
    for i in diar:
        if i['show']==segment['show'] and i['cluster']==segment['cluster'] and i['cluster_type']==segment['cluster_type'] and i['start']==segment['start'] and i['stop']==segment['stop']:
            pass
        else:
            out_diar.append_seg(i)
    return copy.deepcopy(out_diar)

# Checks if the segment is available in the diar object
def existSegment(segment,diar):
    assert isinstance(segment,Segment) and isinstance(diar,Diar)
    return (0!=len(diar.filter("show",'==',segment['show']).filter("cluster","==",segment['cluster']).filter('cluster_type',"==",segment['cluster_type']).filter('start',"==",segment['start']).filter('stop',"==",segment['stop'])))

# Checks if only the segment boundaries are available in the diar object
## WARNING: Don't care of the other information
def existSegmentBoundary(segment,diar):
    assert isinstance(segment,Segment) and isinstance(diar,Diar)
    return (0!=len(diar.filter("start",'==',segment['start']).filter('stop',"==",segment['stop'])))

# Returns a diar object where the spaces of the diar object parameter are filled with the cluster parameter
def fillDiar(cluster,accordingToDiar,diar):
    assert isinstance(cluster,str) and isinstance(diar,Diar) and len(diar)!=0 and len(diar.unique('show'))==1 and isinstance(accordingToDiar,Diar) and accordingToDiar.duration()!=0
    if len(diar.filter('cluster','==',cluster))!=0:
        logging.warning("The cluster parameter is already present in the diar parameter.")
    diarTmp=Diar()
    for i in accordingToDiar:
        diarTmp.append(show=diar[0]['show'], cluster=cluster, start=i['start'], stop=i['stop'])
    diarTmp.pack()
    diarTmp2=copy.deepcopy(diarTmp)
    for j in diarTmp:        
        for i in diar:
            if Segment.intersection(i,j) is not None:
                diarTmp2=releaseFramesFromSegment(i,diarTmp2)
    out_diar=copy.deepcopy(diar)
    out_diar.append_diar(diarTmp2)
    out_diar.sort()
    return out_diar

# Returns the first instervention of a cluster in a diar object, None otherwise           
def firstIntervention(diar,cluster):
    assert isinstance(diar,Diar) and isinstance(cluster,str)
    for i in range(0,len(diar)):
        if diar[i]['cluster']==cluster:
            return i
    return None

# Returns a dict object representing the False Rejection (FR), False Acceptance (FA) and the Match (MATCH)
## tolerance: In centiseconds
## cluster: Cluster name given for a segment absence
def FRFA(diarHyp,diarRef,diarUem=None,tolerance=25,cluster="fillFRFA"):
    assert isinstance(cluster,str) and (diarUem is None or isinstance(diarUem,Diar)) and isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and isinstance(tolerance,numbers.Number)
    tolerance=(tolerance/100)*2
    diarHyp=copy.deepcopy(diarHyp)
    diarRef=copy.deepcopy(diarRef)
    segPre=pyaseg.SegmentationPrecision(tolerance)
    if diarUem is not None:
        diarUem=copy.deepcopy(diarUem)
        diarRef=releaseFramesAccordingToDiar(diarRef,diarUem)
        diarHyp=releaseFramesAccordingToDiar(diarHyp,diarUem)
        diarFill=diarUem
    else:
        maxF=max([max(diarRef.unique('stop')),max(diarHyp.unique('stop'))])
        minF=min([min(diarRef.unique('start')),min(diarHyp.unique('start'))])
        diarFill=Diar()
        diarFill.append(start=minF,stop=maxF)
    diarRef=DiarTools.fillDiar(cluster,diarFill,diarRef)
    diarHyp=DiarTools.fillDiar(cluster,diarFill,diarHyp)
    reference = pyacore.Timeline()
    for i in diarRef:
        reference.add(pyacore.Segment(i['start']/100, i['stop']/100))
    hypothesis = pyacore.Timeline()
    for i in diarHyp:
        hypothesis.add(pyacore.Segment(i['start']/100, i['stop']/100))
    h=(segPre.compute_components(reference=reference, hypothesis=hypothesis))
    r=(segPre.compute_components(reference=reference, hypothesis=reference))
    d=dict()
    d['FA']=h['number of boundaries']-h['number of matches']
    d['FR']=r['number of boundaries']-h['number of matches']
    if d['FA']<0:
        d['FA']=0
    if d['FR']<0:
        d['FR']=0
    d['MATCH']=h['number of matches']
    return d

# Returns a dict object with the human corrections to do on the diarHyp parameter according to the diarRef and diarUem parameters
## WARNING: Doesn't take the labels and a tolerance into consideration
## Note: Similar to the automaton methods
def humanCorrectionsToDo(diarHyp,diarRef,diarUem=None):
    assert isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and (isinstance(diarUem,Diar) or diarUem is None) and len(diarOverlapArea(diarRef))==0 and len(diarOverlapArea(diarHyp))==0 
    diarHyp=copy.deepcopy(diarHyp)
    diarRef=copy.deepcopy(diarRef)
    if diarUem is not None:
        diarUem=copy.deepcopy(diarUem)
        diarRef=releaseFramesAccordingToDiar(diarRef,diarUem)
        diarHyp=releaseFramesAccordingToDiar(diarHyp,diarUem)
    boundHyp=set(diarHyp.unique("stop")).union(set(diarHyp.unique("start")))
    boundRef=set(diarRef.unique("stop")).union(set(diarRef.unique("start")))
    boundToAdd=boundRef.difference(boundHyp)
    boundToDrop=boundHyp.difference(boundRef)
    
    idxRef=diarRef.features_by_cluster()
    idxHyp=diarHyp.features_by_cluster()
    intervalRef=list()
    intervalHyp=list()
    for i in idxRef:
        intervalRef=intervalRef+idxRef[i]
    for i in idxHyp:
        intervalHyp=intervalHyp+idxHyp[i]
    intervalRef.sort()
    intervalHyp.sort()
    intervalExcep=list(set(intervalHyp)-set(intervalRef))
    diarExcep=Diar()
    for i in intervalExcep:
        diarExcep.append(show="show",cluster="cluster",start=i, stop=i+1)
    diarExcep.pack()
    diarChange=copy.deepcopy(diarHyp)
    diarChange=projectBoundaries(diarChange,diarRef)
    for i in diarExcep:
        diarChange=releaseFramesFromSegment(i,diarChange)
    dictio=dict()
    affected=list()
    diarRef.sort()
    diarChange.sort()
    diarCreateOut=Diar()
    diarChangeOut=Diar()
    diarCreateOut.add_attribut('matchCluster', None)
    diarChangeOut.add_attribut('toCluster', None)
    cptNewName=0
    for i in diarRef:
        for y in diarChange:
            inter=Segment.intersection(i,y)
            if inter is not None:
                if i['cluster'] not in dictio:
                    #CREATE
                    if y['cluster'] not in affected:
                        dictio[i['cluster']]=y['cluster']
                        affected.append(y['cluster'])
                        diarCreateOut.append(show=y['show'],
                                             cluster=y['cluster'],
                                             cluster_type=y['cluster_type'],
                                             start=y['start'],
                                             stop=y['stop'],
                                             matchCluster=i['cluster'])
                    else:
                        dictio[i['cluster']]="None#"+str(cptNewName)
                        diarCreateOut.append(show=y['show'],
                                             cluster=y['cluster'],
                                             cluster_type=y['cluster_type'],
                                             start=y['start'],
                                             stop=y['stop'],
                                             matchCluster="None#"+str(cptNewName))
                        cptNewName+=1
                else:
                    if dictio[i['cluster']]==y['cluster']:
                        #NOTHING
                        pass
                    else:
                        #CHANGE
                        diarChangeOut.append(show=y['show'],
                                             cluster=y['cluster'],
                                             cluster_type=y['cluster_type'],
                                             start=y['start'],
                                             stop=y['stop'],
                                             toCluster=dictio[i['cluster']])
    
    for i in diarExcep:
        for y in diarHyp:
            j=copy.deepcopy(y)
            j['show']=i['show']
            j['cluster']=i['cluster']
            if Segment.intersection(i,j) is not None:
                i['show']=y['show']
                i['cluster']=y['cluster']
                i['cluster_type']=y['cluster_type']
    diarExcep.add_attribut('toCluster',"Non_Speech")
    diarChangeOut.append_diar(diarExcep)
    diarChangeOut.sort()
    dictToReturn=dict()
    dictToReturn["createLabel"]=diarCreateOut
    dictToReturn["changeLabel"]=diarChangeOut
    dictToReturn["createBoundary"]=boundToAdd
    dictToReturn["dropBoundary"]=boundToDrop
    return dictToReturn

# Returns the HCIQ value
def HCIQ(nCb,nDb,nCn,nSn,nV,cb=12,db=5.1,cn=12.7,sn=7.6,v=0):
    return nCb*cb+nDb*db+nCn*cn+nSn*sn+nV*v

# Returns the matching segments with a given segment, None otherwise
def matchingSegmentsFromSegment(segment,diar):
    assert isinstance(diar,Diar) and isinstance(segment,Segment)
    dictMatch=dict()
    diar=copy.deepcopy(diar)
    diar.sort()
    ind=False
    for i in diar:
        inter=Segment.intersection(segment,i)
        if inter is not None:
            ind=True
            inter['show']=i['show']
            inter['cluster']=i['cluster']
            inter['cluster_type']=i['cluster_type']

            if i['cluster'] not in dictMatch:
                dictMatch[i['cluster']]=Diar()
                dictMatch[i['cluster']].append_seg(inter)
            else:
                dictMatch[i['cluster']].append_seg(inter)
        elif ind:
            break
    return dictMatch

# Returns a new diar object corresponding to the merging of two diar objects
def mergeDiar(diarBase,diarAdd,sort=False):
    assert isinstance(diarBase,Diar) and isinstance(diarAdd,Diar) and isinstance(sort,bool)
    out_diar=copy.deepcopy(diarBase)
    add_diar=copy.deepcopy(diarAdd)
    out_diar.append_diar(diarAdd)
    if sort:
        out_diar.sort()
    return out_diar

# Returns a diar object with two segments merged
def mergeSegment(segment1,segment2,diar):
    assert isinstance(segment1,Segment) and isinstance(segment2,Segment) and isinstance(diar,Diar)
    assert segment1['show']==segment2['show'] and segment1['cluster']==segment2['cluster'] and segment1['cluster_type']==segment2['cluster_type']
    assert Segment.intersection(segment1,segment2) is not None or segment1["stop"]==segment2["start"] or segment1["start"]==segment2["stop"]
    assert countOccurenceSegment(segment1,diar)==1 and countOccurenceSegment(segment2,diar)==1
    out_diar=dropSegment(segment2,dropSegment(segment1,diar))
    if segment1['start']<=segment2['start']:
        start=segment1['start']
    else:
        start=segment2['start']
    if segment1['stop']>=segment2['stop']:
        stop=segment1['stop']
    else:
        stop=segment2['stop']
    segment=copy.deepcopy(segment1)
    segment['start']=start
    segment['stop']=stop  
    out_diar.append_seg(segment)
    out_diar.sort()
    return [copy.deepcopy(segment),out_diar]

# Returns a diar object with the segment put in parameter transformed/moved at new boundaries
def moveSegment(segment,diar,start,stop):
    assert isinstance(segment,Segment) and isinstance(diar,Diar)
    assert isinstance(start,numbers.Number) or start is None
    assert isinstance(stop,numbers.Number) or stop is None
    if start is not None and stop is not None:
        assert start<stop
    out_diar=Diar()
    for i in diar:
        if i['show']==segment['show'] and i['cluster']==segment['cluster'] and i['cluster_type']==segment['cluster_type'] and i['start']==segment['start'] and i['stop']==segment['stop']:
            iTmp=copy.deepcopy(i)
            if start is not None:
                assert start<iTmp['stop']
                iTmp['start']=start
            if stop is not None:
                assert stop>iTmp['start']
                iTmp['stop']=stop
            out_diar.append_seg(iTmp)
        else:
            out_diar.append_seg(i)
    return copy.deepcopy(out_diar)

# Checks if there are no gaps within the given diar
def noGapWithinDiar(diar):
    assert isinstance(diar,Diar)
    minTmp=min(diar.unique('start'))
    maxTmp=max(diar.unique('stop'))
    diarTmp=Diar()
    diarTmp.append(start=minTmp,stop=maxTmp)
    for i in diar:
        diarTmp=releaseFramesFromSegment(i,diarTmp)
    if len(diarTmp)==0:
        return True
    return False

# Returns a diar object with boundaries from a diar object projected to another one
def projectBoundaries(diar,diarSource):
    assert isinstance(diar,Diar) and isinstance(diarSource,Diar)
    outputDiar=Diar()
    for i in diarSource:
        dictio=matchingSegmentsFromSegment(i,diar)
        for j in dictio:
            for y in dictio[j]:
                y['cluster']=j
                outputDiar.append_seg(y)
    diarClone=copy.deepcopy(diar)
    for i in outputDiar:
        diarClone=releaseFramesFromSegment(i,diarClone)
    outputDiar.append_diar(diarClone)
    outputDiar.sort()
    return outputDiar

# Returns a dict object representing the Purity and the Coverage
## tolerance: In centiseconds
## skip_overlap: Computing according to overlap or not
def PurityCoverage(diarHyp,diarRef,diarUem=None,tolerance=25,skip_overlap=False):
    assert isinstance(skip_overlap,bool) and (diarUem is None or isinstance(diarUem,Diar)) and isinstance(diarHyp,Diar) and isinstance(diarRef,Diar) and isinstance(tolerance,numbers.Number)
    tolerance=(tolerance/100)*2

    reference = pyacore.Annotation()
    for seg in diarRef:
        s = pyacore.Segment(seg['start'], seg['stop'])
        if s not in reference:
            reference[s, 1] = seg['cluster']
        else:
            reference[s, 2] = seg['cluster']

    if diarUem is None:
        ref_uem=None
    else:
        ref_uem = pyacore.Timeline()
        for seg in diarUem:
            ref_uem.add(pyacore.Segment(seg['start'], seg['stop']))

    hyp = pyacore.Annotation()
    for seg in diarHyp:
        hyp[pyacore.Segment(seg['start'], seg['stop'])] = seg['cluster']

    p=pyadiar.DiarizationPurity(collar=tolerance,skip_overlap=skip_overlap)
    c=pyadiar.DiarizationCoverage(collar=tolerance,skip_overlap=skip_overlap)
    rtn=dict()
    rtn["Purity"]=p.compute_metric(p.compute_components(reference,hyp,ref_uem))*100
    rtn["Coverage"]=c.compute_metric(c.compute_components(reference,hyp,ref_uem))*100
    return rtn

# Reads any file according the extension
def read(fn):
    if isinstance(fn,str):
        dir, name, ext = path_show_ext(fn)
        if ext.endswith('.seg'):
            diar = Diar.read_seg(fn)
        elif ext == '.mdtm':
            diar = Diar.read_mdtm(fn)
        elif ext == '.rttm':
            diar = Diar.read_rttm(fn)
        elif ext == '.uem':
            diar = Diar.read_uem(fn)
        else:
            logging.error('Unknown format: '+ext)
            raise Exception('Wrong format.')
        shows = diar.unique('show')
        shows.sort()
        clusters = diar.unique('cluster')
        clusters.sort()
        return diar, shows, clusters
    else:
        logging.error("The parameter has to be a str type.")
        raise Exception('Wrong type.')

# Returns a diar object without the frames between start and stop
## start: In centiseconds
## stop: In centiseconds
def releaseFrames(start,stop,diar):
    assert isinstance(start,numbers.Number) and isinstance(stop,numbers.Number) and isinstance(diar,Diar)
    segment=dict()
    segment['start']=start
    segment['stop']=stop
    out_diar=Diar()
    for row in diar:
        if (row['start']>=segment['start'] and row['start']<segment['stop']) or (
            row['stop']>segment['start'] and row['stop']<=segment['stop']):
            if (row['start']>=segment['start'] and row['start']<segment['stop']) and (
                row['stop']>segment['start'] and row['stop']<=segment['stop']):
                    pass
            elif (row['start']>=segment['start'] and row['start']<segment['stop']):
                rowTmp=copy.deepcopy(row)
                rowTmp['start']=segment['stop']
                out_diar.append_seg(rowTmp)
            else:
                rowTmp=copy.deepcopy(row)
                rowTmp['stop']=segment['start']
                out_diar.append_seg(rowTmp)
        elif row['start']<segment['start'] and row['stop']>segment['stop'] :
            rowTmp1=copy.deepcopy(row)
            rowTmp2=copy.deepcopy(row)
            rowTmp1['stop']=segment['start']
            rowTmp2['start']=segment['stop']
            out_diar.append_seg(rowTmp1)
            out_diar.append_seg(rowTmp2)
        else:
            out_diar.append_seg(copy.deepcopy(row))
    return out_diar

# Returns a diar object without the frames not defined in a diar object put in parameter
def releaseFramesAccordingToDiar(diar,basisDiar):
    assert isinstance(diar,Diar) and isinstance(basisDiar,Diar)
    out_diar=copy.deepcopy(diar)
    maxLen=max(basisDiar.unique('stop'))
    maxLenTmp=max(diar.unique('stop'))
    if maxLenTmp>maxLen:
        maxLen=maxLenTmp
    maxLen+=100
    diarTmp=Diar()
    diarTmp.append(start=0,stop=maxLen)
    for i in copy.deepcopy(basisDiar):
        diarTmp=releaseFramesFromSegment(i,diarTmp)
    for i in diarTmp:
        out_diar=releaseFramesFromSegment(i,out_diar)
    return out_diar

# Returns a diar object with releasing all the frames not defined in the input diar object and its tolerance boundaries
## basisTolerance: In centiseconds
def releaseFramesAccordingToDiarWithToleranceBoundaries(diar,basisDiar=None,basisTolerance=0):
    assert isinstance(diar,Diar) and (basisDiar is None or isinstance(basisDiar,Diar)) and isinstance(basisTolerance,numbers.Number)
    if basisDiar is None:
        basisDiar=diar
    basisTolerance=abs(basisTolerance)
    out_diar=copy.deepcopy(diar)
    maxLen=max(basisDiar.unique('stop'))
    maxLenTmp=max(diar.unique('stop'))
    if maxLenTmp>maxLen:
        maxLen=maxLenTmp
    maxLen+=100
    diarTmp=Diar()
    diarTmp.append(start=0,stop=maxLen)
    for i in copy.deepcopy(basisDiar):
        diarTmp=releaseFramesFromSegment(i,diarTmp)
        if segmentExistAccordingToTolerance(i,basisTolerance):
            out_diar=releaseFrames(i['start']-basisTolerance,i['start']+basisTolerance,out_diar)
            out_diar=releaseFrames(i['stop']-basisTolerance,i['stop']+basisTolerance,out_diar)
        else:
            i['start']-=basisTolerance
            i['stop']+=basisTolerance
            out_diar=releaseFramesFromSegment(i,out_diar) 
    for i in diarTmp:
        out_diar=releaseFramesFromSegment(i,out_diar)
    return out_diar

# Returns a diar object without the frames according to the position start for a duration of time
## start: In centiseconds
## time: In centiseconds
def releaseFramesAccordingToTime(start,time,diar):
    return releaseFrames(start=start,stop=start+time,diar=diar)

# Returns a diar object without the frames matching the segment put in parameter
def releaseFramesFromSegment(segment,diar):
    assert isinstance(segment,Segment) and isinstance(diar,Diar)
    assert segment['start']<=segment['stop']
    out_diar=Diar()
    for row in diar:
        if (row['start']>=segment['start'] and row['start']<segment['stop']) or (
            row['stop']>segment['start'] and row['stop']<=segment['stop']):
            if (row['start']>=segment['start'] and row['start']<segment['stop']) and (
                row['stop']>segment['start'] and row['stop']<=segment['stop']):
                    pass
            elif (row['start']>=segment['start'] and row['start']<segment['stop']):
                rowTmp=copy.deepcopy(row)
                rowTmp['start']=segment['stop']
                out_diar.append_seg(rowTmp)
            else:
                rowTmp=copy.deepcopy(row)
                rowTmp['stop']=segment['start']
                out_diar.append_seg(rowTmp)
        elif row['start']<segment['start'] and row['stop']>segment['stop'] :
            rowTmp1=copy.deepcopy(row)
            rowTmp2=copy.deepcopy(row)
            rowTmp1['stop']=segment['start']
            rowTmp2['start']=segment['stop']
            out_diar.append_seg(rowTmp1)
            out_diar.append_seg(rowTmp2)
        else:
            out_diar.append_seg(copy.deepcopy(row))
    return out_diar

# Returns a diar object instance without some segments in accordance with the duration and the restrictions put in parameter
## duration: In centiseconds
def restrictSegments(diar,duration,above=False,equal=False):
    assert isinstance(diar,Diar) and isinstance(duration,numbers.Number) and isinstance(above,bool) and isinstance(equal,bool)
    diarFinal=Diar()
    for i in diar:
        if equal:
            if above:
                if i.duration() < duration:
                    diarFinal.append_seg(i)
            else:
                if i.duration() > duration:
                    diarFinal.append_seg(i)
        else:
            if above:
                if i.duration() <= duration:
                    diarFinal.append_seg(i)
            else:
                if i.duration() >= duration:
                    diarFinal.append_seg(i)
    return copy.deepcopy(diarFinal)

# Returns a reverse diar object
## maxStop: Optional (by default we use the stop max from the given diar). In centiseconds
def reverseDiar(diar,cluster="NOISE",maxStop=None):
    assert isinstance(diar,Diar) and (maxStop is None or isinstance(maxStop,numbers.Number)) and len(diar.unique('show'))==1 and isinstance(cluster,str)
    if maxStop is not None and maxStop<=0:
        return Diar()
    diarTmp=Diar()
    if maxStop is not None:
        diarTmp.append(show=diar.unique('show')[0],cluster=cluster,start=0,stop=maxStop)
    else:
        diarTmp.append(show=diar.unique('show')[0],cluster=cluster,start=0,stop=max(diar.unique('stop')))
    for i in diar:
        diarTmp=releaseFrames(i['start'],i['stop'],diar=diarTmp)
    return diarTmp

# Returns a diar object by removing overlapped segments (same start & stop) lower than epsilon and by moving the boundaries of the previous or the next segment if the names correspond to the dropped segments
## epsilon: In centiseconds
def rmOvlpSegments(diar, epsilon):
    assert isinstance(diar,Diar) and isinstance(epsilon,numbers.Number)
    for i in range(1, len(diar)-2):
        seg0 = diar[i-1]
        seg1 = diar[i]
        seg2 = diar[i+1]
        seg3 = diar[i+2]
        print(seg1['start'], seg2['start'], seg1['start'] == seg2['start'],
                 seg1['stop'] == seg2['stop'],
                 seg1.duration() <= epsilon)
        if seg1['start'] == seg2['start'] \
                and seg1['stop'] == seg2['stop'] \
                and seg1.duration() <= epsilon :
            l = seg2.duration()
            d = seg2.duration() / 2
            seg1['start'] += d
            seg1['stop'] -= d
            seg2['start'] += d
            seg2['stop'] -= d
            names = set([seg1['cluster'], seg2['cluster']])
            if seg0['cluster'] != seg3['cluster']:
                if seg0['cluster'] in names and seg3['cluster'] in names :
                    seg0['stop'] += d
                    seg3['start'] -= d
                elif seg0['cluster'] in names:
                    seg0['stop'] += l
                elif seg3['cluster'] in names:
                    seg3['start'] -= l

    diar = diar.filter('duration', '>', 0)
    diar.pack(100)

    return diar

# Returns a diar object with safe annotation according to the diarUem parameter 
## WARNING: Without overlapped segments or speech turn < "withoutSpeechTurn" or the number of followed segments < "segmentFollowed"
## withoutSpeechTurn: In centiseconds
def safeAnnotationDiar(diar,diarUem=None,addNoise=True,segmentFollowed=None,withoutSpeechTurn=100):
    assert isinstance(withoutSpeechTurn,numbers.Number) and isinstance(diar,Diar) and (isinstance(diarUem,Diar) or diarUem is None) and (isinstance(segmentFollowed,numbers.Number) or segmentFollowed is None) and isinstance(addNoise,bool)
    diar=copy.deepcopy(diar)
    if diarUem is None:
        diarUem=Diar()
        diarUem.append(show=diar[0]["show"], cluster='nonspeech', start=min(diar.unique('start')), stop=max(diar.unique('stop')))
    else:
        diarUem=copy.deepcopy(diarUem)
    diarNoise=copy.deepcopy(diarUem)
    for i in diar:
        diarNoise=releaseFramesFromSegment(i,diarNoise)
    diarOverlap=diarOverlapArea(diar)
    for i in diarOverlap:
        diar=releaseFramesFromSegment(i,diar)
    for i in diar:
        if i.duration()<=withoutSpeechTurn:
            diar=releaseFramesFromSegment(i,diar)
    if addNoise:
        diar=mergeDiar(diar,diarNoise,True)
    diarOut=Diar()
    diarTmp=Diar()
    counter=0
    for idx,i in enumerate(diar):
        if i["start"]==diar[idx-1]["stop"]:
            diarTmp.append_seg(i)
            if (addNoise and not existSegment(i,diarNoise)) or not addNoise:
                counter+=1
            if idx==1:
                diarTmp.append_seg(diar[0])
                if (addNoise and not existSegment(i,diarNoise)) or not addNoise:
                    counter+=1
        else:
            if (segmentFollowed is not None and counter>=segmentFollowed) or segmentFollowed is None:
                diarOut=mergeDiar(diarOut,diarTmp,True)
            counter=0
            diarTmp=Diar()
    for i in diarOut:
        i["cluster"]="safeArea"
    diarOut.pack()
    return diarOut

# Checks if the segment is greater than tolerance*2 (boundary with tolerance spaces)
## tolerance: In centiseconds
def segmentExistAccordingToTolerance(segment,tolerance):
    assert isinstance(segment,Segment) and isinstance(tolerance,numbers.Number)
    tolerance=abs(tolerance)
    if segment.duration()>tolerance*2:
        return True
    return False

# Checks if the given segment is within the given diar object
def segmentWithinDiar(segment,diar):
    assert isinstance(segment,Segment) and isinstance(diar,Diar)
    diar=copy.deepcopy(diar)
    for i in diar:
        i['cluster']="test"
    diar.pack()
    diar.sort()
    for i in diar:
        inter=Segment.intersection(segment,i)
        if inter is not None and inter.duration()==segment.duration():
            return True
    return False

# Returns a diar object with a specific segment splitted into two according to the parameter "value"
def splitSegment(segment,diar,value):
    assert isinstance(segment,Segment) and isinstance(diar,Diar) and isinstance(value,numbers.Number) and value>segment['start'] and value<segment['stop']
    out_diar=Diar()
    for i in diar:
        if i['show']==segment['show'] and i['cluster']==segment['cluster'] and i['cluster_type']==segment['cluster_type'] and i['start']==segment['start'] and i['stop']==segment['stop']:
            out_diar.append(show=segment['show'], cluster=segment['cluster'],cluster_type=segment['cluster_type'], start=segment['start'], stop=value)
            out_diar.append(show=segment['show'], cluster=segment['cluster'],cluster_type=segment['cluster_type'], start=value, stop=segment['stop'])           
        else:
            out_diar.append_seg(i)
    return copy.deepcopy(out_diar)

# Returns a vector v[i]:x (a dict object) from a diar object where "i" is the trame and "x" the related name
def vector(diar,start,stop):
    assert isinstance(diar,Diar) and isinstance(start,numbers.Number) and isinstance(stop,numbers.Number)
    diarTmp=Diar()
    diarTmp.append(show='tmp', cluster='name', start=start, stop=stop)
    diar=copy.deepcopy(diar)
    diar=releaseFramesAccordingToDiar(diar,diarTmp)
    diarIdx=diar.features_by_cluster()
    dictGift=dict()
    for i in diarIdx:
        for j in diarIdx[i]:
            dictGift[j]=i
    return dictGift

