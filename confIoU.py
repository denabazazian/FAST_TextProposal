#!/usr/bin/env python
#import sklearn.naive_bayes#


from multiprocessing import Pool
import os.path

import numpy as np #
from commands import getoutput as go
import sys
import time
import numpy.matlib #
from collections import defaultdict
import cPickle
import scipy.stats#

import cv2#
import matplotlib.pyplot as plt#


def myread(fname):
    txt=open(fname).read().strip()
    if txt[:3]=='\xef\xbb\xbf':
        txt=txt[3:]
    return txt

def download(url,fname):
    cmd='wget -c '+url+' -O '+fname
    go(cmd)





def get2PointIU(gtMat,resMat):
    gtMat=gtMat.copy()
    resMat=resMat.copy()
    maxProposalsIoU=int(switches['maxProposalsIoU'])
    if maxProposalsIoU>0:
        resMat=resMat[:maxProposalsIoU,:]
    #matSz=(gtMat.shape[0],resMat.shape[0])
    gtLeft=numpy.matlib.repmat(gtMat[:,0],resMat.shape[0],1)
    gtTop=numpy.matlib.repmat(gtMat[:,1],resMat.shape[0],1)
    gtRight=numpy.matlib.repmat((gtMat[:,0]+gtMat[:,2])-1,resMat.shape[0],1)
    gtBottom=numpy.matlib.repmat((gtMat[:,1]+gtMat[:,3])-1,resMat.shape[0],1)
    gtWidth=numpy.matlib.repmat(gtMat[:,2],resMat.shape[0],1)
    gtHeight=numpy.matlib.repmat(gtMat[:,3],resMat.shape[0],1)

    resLeft=numpy.matlib.repmat(resMat[:,0],gtMat.shape[0],1).T
    resTop=numpy.matlib.repmat(resMat[:,1],gtMat.shape[0],1).T
    resRight=numpy.matlib.repmat((resMat[:,0]+resMat[:,2])-1,gtMat.shape[0],1).T
    resBottom=numpy.matlib.repmat((resMat[:,1]+resMat[:,3])-1,gtMat.shape[0],1).T
    resWidth=numpy.matlib.repmat(resMat[:,2],gtMat.shape[0],1).T
    resHeight=numpy.matlib.repmat(resMat[:,3],gtMat.shape[0],1).T

    intL=np.max([resLeft,gtLeft],axis=0)
    intT=np.max([resTop,gtTop],axis=0)

    intR=np.min([resRight,gtRight],axis=0)
    intB=np.min([resBottom,gtBottom],axis=0)

    intW=(intR-intL)+1
    intW[intW<0]=0

    intH=(intB-intT)+1
    intH[intH<0]=0

    I=intH*intW
    U=resWidth*resHeight+gtWidth*gtHeight-I
    minsurf=np.minimum(resHeight*resWidth,gtWidth*gtHeight)
    IoU=I/(U+.0000000001)
    resTLRB=resMat[:,:]
    resTLRB[:,[2,3]]+=resTLRB[:,[0,1]]-1
    gtTLRB=gtMat[:,:]
    gtTLRB[:,[2,3]]+=gtTLRB[:,[0,1]]-1
    return (IoU,I,U,minsurf)

#filename conversions
def createRequiredDirs(filenameList,fromFilesDir):
    filesDirList=set(['/'.join(f.split('/')[:-1]) for f in filenameList])
    if fromFilesDir[0]=='+':
        for fd in filesDirList:
            addP=fromFilesDir[1:].split('/')
            addP[-1]=addP[-1]+fd.split('/')[-1]
            go('mkdir -p '+fd+'/'+'/'.join(addP))
    else:
        for fd in filesDirList:
            go('mkdir -p '+fd+'/'+fromFilesDir)


def getInputFromConf(confFname):
    pathList=confFname.split('/')
    pathList[-2]='input'
    return '/'.join(pathList)[:-3]+'jpg'


def getThresholdFromHm(hmFname,outDir):
    pathList=hmFname.split('/')
    pathList[-2]=outDir+pathList[-2]
    return '/'.join(pathList)[:-3]+'csv'


def getProposalFromConf(hmFname):
    pathList=hmFname.split('/')
    pathList[-2]=switches['propDir']
    return ('/'.join(pathList)) [:-3]+'csv'


def getConfFromHm(hmFname):
    pathList=hmFname.split('/')
    pathList[-2]='conf_'+pathList[-2]
    return '/'.join(pathList)[:-3]+'csv'


def getIouFromConf(confFname):
    pathList=confFname.split('/')
    pathList[-2]='iou_'+pathList[-2]
    return '/'.join(pathList)[:-3]+'png'


def getGtFromConf(imageFname):
    pathList=imageFname.split('/')
    pathList[-2]='gt'
    return '/'.join(pathList)[:-3]+'txt'


def getProposalFromImage(imageFname,thr=0,hm=''):
    if thr==0:
        pathList=imageFname.split('/')
        pathList[-2]=switches['propDir']
        return '/'.join(pathList)[:-3]+'csv'
    else:
        pathList=imageFname.split('/')
        pathList[-2]='prop%d%s'%(int(thr*100),hm)
        return '/'.join(pathList)[:-3]+'csv'


def getProposalFromHeatmap(heatmapFname):
    if heatmapFname[-3:]=='png':
        heatmapFname=heatmapFname[:-3]+'csv'
    pathList=heatmapFname.split('/')
    pathList[-2]=switches['propDir']
    return '/'.join(pathList)

def getConfidenseFromHeatmap(heatmapFname):
    if heatmapFname[-3:]=='png':
        heatmapFname=heatmapFname[:-3]+'csv'
    pathList=heatmapFname.split('/')
    pathList[-2]='conf_'+pathList[-2]
    return '/'.join(pathList)

getConfidenseFromProposal=getConfidenseFromHeatmap

#str 2 numpy
def arrayToCsvStr(arr):
    if len(arr.shape)!=2:
        raise Exception("only 2D arrays")
    resLines=[list(row) for row in list(arr)]
    return '\n'.join([','.join([str(col) for col in row])  for row in resLines])

def csvStr2Array(csvStr):
    return np.array([[float(c) for c in l.split(',')] for l in csvStr.split('\n') if len(l)>0])

def fname2Array(fname):
    if fname[-3:]=='png':
        return cv2.imread(fname,cv2.IMREAD_GRAYSCALE)/255.0
    else: #assuming csv
        if os.path.isfile(fname) and os.path.getsize(fname)>0:
            res= np.genfromtxt(fname, delimiter=',')
            return res
        else:
            return np.empty([0,5])


def array2csvFname(arr,csvFname):
    np.savetxt(csvFname,arr, '%9.5f',delimiter=',')

def array2pngFname(arr,pngFname):
    cv2.imwrite(pngFname,(arr*255).astype('uint8'),[cv2.IMWRITE_PNG_COMPRESSION ,0])


def loadTxtGtFile(fname):
    txt=myread(fname)
    lines=[l.strip().split(',') for l in txt.split('\n') if len(l.strip())>0]
    print '{%s}'%lines
    if len(lines)==0:
        return np.zeros([0,4]),np.zeros([0],dtype=object)
#    if len(lines)==0:
#        return (np.array([5,5,3,3],dtype='float'),['###'])
#    if lines[0][0][:3]=='\xef\xbb\xbf':
#        lines[0][0]=lines[0][0][3:]

#Raul: Remove bugged lines

    for line in lines:
        if len(line) < 4:
            print 'BAD LINE:',str(line),'\nFile:',fname
            raise Exception()
            lines.remove(line)
        elif line == ['|']:
            lines.remove(line)

    for line in lines:
        if len(line) < 4:
            lines.remove(line)
        elif line == ['|']:
            lines.remove(line)

    if len(lines[0])>8:#4 points
        rects=np.empty([len(lines),4],dtype='float')
        tmpArr=np.array([[int(c) for c in line[:8]] for line in lines])
        left=tmpArr[:,[0,2,4,6]].min(axis=1)
        right=tmpArr[:,[0,2,4,6]].max(axis=1)
        top=tmpArr[:,[1,3,5,7]].min(axis=1)
        bottom=tmpArr[:,[1,3,5,7]].max(axis=1)
        rects[:,0]=left
        rects[:,1]=top
        rects[:,2]=1+right-left
        rects[:,3]=1+bottom-top
        trans=[','.join(line[8:]) for line in lines]
    else:#ltwh
        rects=np.array([[int(c) for c in line[:4]] for line in lines],dtype='float')
        trans=[','.join(line[4:]) for line in lines]
    return (rects,trans)


def loadVggTranscription(fname):
    txt=myread(fname)
    lines=[l.split(',') for l in txt.split('\n')]
    LTWHConf=np.empty([len(lines),5])
    transcriptions=np.empty([LTWHConf.shape[0]],dtype='object')
    transcriptions[:]=[l[-1] for l in lines]
    if len(lines) > 1:
    	LTWHConf[:,:]=np.array([[float(c) for c in l[:-1]] for l in lines])
    return LTWHConf,transcriptions

def getDontCare(transcriptions,dictionary=[]):
    dictionary=set(dictionary)
    if len(dictionary)==0:
        return np.array([tr!='###' for tr in transcriptions],dtype='bool')
    else:
        return np.array([(tr in dictionary) for tr in transcriptions],dtype='bool')


def getNmsIdx(LTWHConf):
    #sort rectangles from large to small
    #LTWHConf=LTWHConf[np.argsort(LTWHConf[:,4].prod(axis=1))[::-1],:]
    (IoU,I,U,minsurf)=get2PointIU(LTWHConf[:,:4],LTWHConf[:,:4])
    iouThr=eval(switches['iouThr'])
    return np.where((np.triu(IoU,1)>iouThr).sum(axis=0)<1)[0]
    #return LTWHConf[(np.triu(IoU,1)>iouThr).sum(axis=0)<1,:]


#algorithm
def getConfidenceForAll(hm,prop):
    prop=prop.astype('int32')
    ihm=np.zeros([hm.shape[0]+1,hm.shape[1]+1])
    ihm[1:,1:]=hm.cumsum(axis=0).cumsum(axis=1) #integral image
    confidenceDict=defaultdict(list)
    for rectId in range(prop.shape[0]):
        rect=tuple(prop[rectId,:4])
        (l,t,w,h)=rect
        r=l+w#the coordinates are translated by 1 because of ihm zero pad
        b=t+h
        confidenceDict[rect].append(((ihm[b,r]+ihm[t,l])-(ihm[b,l]+ihm[t,r]))/(w*h))
    res=np.array([tup[1]+(tup[0],) for tup in sorted([(max(confidenceDict[rec]),rec) for rec in confidenceDict.keys()],reverse=True)])
    return res


def postProcessProp(propCSV):
    lines=[tuple([float(c) for c in  l.split(',')]) for l in  propCSV.split('\n') if len(l)>0]
    rectDict=defaultdict(list)
    for l in lines:
        rectDict[l[:4]].append(l[4:])
    reslines=[]
    for r in rectDict.keys():
        reslines.append(r+max(rectDict[r]))
    if len(reslines):
        proposals=np.array(reslines)
        proposals=proposals[np.argsort(-proposals[:,4]),:]
        proposals=[[int(p[0]),int(p[1]),int(p[2]),int(p[3]),p[4],p[5],int(p[6]),int(p[7]),int(p[8]),int(p[9])] for p in proposals.tolist()]
        return '\n'.join([','.join([str(c) for c in p]) for p in proposals])
    else:
        return ''



switches={'maxProposalsIoU':'200000',#IoU over this are not computed #20000
'gpu':None,
'img2propPath':'./tmp/TextProposalsInitialSuppression/img2hierarchy',
'weakClassifier':'./tmp/TextProposalsInitialSuppression/trained_boost_groups.xml',
'proto':'/tmp/dictnet_vgg_deploy.prototxt',
'pretrained':'/tmp/dictnet_vgg.caffemodel',
'vocabulary':'/tmp/dictnet_vgg_labels.txt',
'dictnetThr':'0.004',
'vocDir':'voc_strong',
'iouThr':'0.3',
'minibatchSz':'100',
'threads':'1',
'dontCareDictFile':'',
'IoUThresholds':'[.5]',
'extraPlotDirs':'{".":"Confidence"}',
'care':'True', #If true dont cares are supressed
'bayesianFname':'/tmp/bayesian.cPickle',
'plotter':'plt.semilogx',
'thr':'0.1',
'plotfname':'plots.pdf',
'fixedProps':'1000',#,
'propDir':'proposals'
}



if __name__=='__main__':
    hlp="""
    hm2conf blabla/*/heatma.../*.csv
    hm2conf blabla/*/heatma.../*.png
    img2prop blabla/*/input/*.jpg
    """
    params=[(len(p)>0 and p[0]!='-',p) for p in sys.argv]
    sys.argv=[p[1] for p in params if p[0]]
    switches.update(dict([p[1][1:].split('=') for p in params if not p[0]]))
    print 'Threads',int(switches['threads'])
    if sys.argv[1]=='dictnet2final':
        #dnThrCount=0
        def worker(vggfname):
            outDir='finalDN%dIou%dVoc%s_%s'%((int(100*eval(switches['dictnetThr']))),(int(100*eval(switches['iouThr']))),switches['vocDir'].split('_')[-1],vggfname.split('/')[-2])
            ofname=vggfname.split('/')
            ofname[-2]=outDir
            ofname='/'.join(ofname)
            LTWHConf,transcriptions=loadVggTranscription(vggfname)

            #FIXING UNSORTED YOLO/TEXTBOXE BEGIN
            LTWHTextnes=fname2Array(vggfname.replace('vggtr_',''))
            LTWHTextnes[LTWHTextnes[:,2]<3,2]=3
            LTWHTextnes[LTWHTextnes[:,3]<3,3]=3
            if set([tuple(l) for l in LTWHTextnes[:,:4].astype('int32').tolist()])!=set([tuple(l) for l in LTWHConf[:,:4].astype('int32').tolist()]):
                print "Count not fincd consistent conf "+vggfname
                raise Exception()
            sortdeterministic=lambda x: (x[:,0]+x[:,1]*(10^4)+x[:,0]*(10^8)+x[:,1]*(10^12)).argsort()
            LTWHTextnes=LTWHTextnes[sortdeterministic(LTWHTextnes),:]#Making the textnes follow a dterministic order
            idx=sortdeterministic(LTWHConf)#alligning vggtr with confidence rectangles
            transcriptions=transcriptions[idx]#alligning vggtr with confidence rectangles
            LTWHConf=LTWHConf[idx,:]#alligning vggtr with confidence rectangles
            #now the 4 first columns of LTWHConf and LTWHTextnes should be the same rectangles
            if LTWHTextnes[:,:4].astype('int32').tolist()!=LTWHConf[:,:4].astype('int32').tolist():
                print "Failed to allign rectangles "+vggfname
                raise Exception()
            textnesSortedIdx=np.argsort(-LTWHTextnes[:,4])
            LTWHConf=LTWHConf[textnesSortedIdx,:]
            transcriptions=transcriptions[textnesSortedIdx]
            #FIXING UNSORTED YOLO/TEXTBOXE END

            print vggfname.split('/')[-2].split('_')[-1]+'/'+vggfname.split('/')[-1],' Initial :',LTWHConf.shape[0],
            filterIdx=LTWHConf[:,4]>eval(switches['dictnetThr'])
            print ' dictnet>'+str(float(int(eval(switches['dictnetThr'])*100))/100)+' kills:',(filterIdx==0).sum(),
            LTWHConf=LTWHConf[filterIdx,:]
            transcriptions=transcriptions[filterIdx]
            #print LTWHConf[:,2:4].max()
            #print '\n\n#2\n','\n'.join(transcriptions.tolist())
            if switches['vocDir'] and LTWHConf.size>0:
                vocFname=ofname.split('/')
                vocFname[-2]=switches['vocDir']
                vocFname='/'.join(vocFname)[:-3]+'txt'
                voc=set([s.lower().strip() for s in myread(vocFname).split('\n')[:-1]])
                filterIdx=np.ones(transcriptions.shape[0],dtype='bool')
                for k in range(transcriptions.shape[0]):
                    filterIdx[k]=transcriptions[k].lower() in voc
                print ' VOC kills:', (filterIdx==0).sum(),
                LTWHConf=LTWHConf[filterIdx,:]
                transcriptions=transcriptions[filterIdx]
            res=[]
            filterIdx=getNmsIdx(LTWHConf)
            print 'NMS kills:', LTWHConf.shape[0]-(filterIdx).shape[0],' SURVIVED:',(filterIdx.shape[0])
            LTWHConf=LTWHConf[filterIdx,:]
            transcriptions=transcriptions[filterIdx]
            for k in range(transcriptions.shape[0]):
                resLine=','.join([str(int(c)) for c in LTWHConf[k,:4]])+','+transcriptions[k]
                #print 'MAX RESLINE:',resLine
                res.append(resLine)
            open(ofname,'w').write('\n'.join(res))
        outDirL= lambda x:'finalDN%dIou%dVoc%s_%s'%((int(100*eval(switches['dictnetThr']))),(int(100*eval(switches['iouThr']))),switches['vocDir'].split('_')[-1],x.split('/')[-2])
        [go('mkdir -p '+d) for d in  set('/'.join(f.split('/')[:-2]+[outDirL(f)]) for f in  sys.argv[2:])]
        if int(switches['threads'])<=1:
            [worker(f) for f in sys.argv[2:]]
        else:
            pool=Pool(int(switches['threads']))
            pool.map(worker,sys.argv[2:])
        sys.exit(0)

    if sys.argv[1]=='gt2SW':
        widths=np.empty([1000000])
        heights=np.empty([1000000])
        objCount=0
        for fname in sys.argv[2:]:
            gtMat,transcriptions=loadTxtGtFile(fname)
            widths[objCount:objCount+gtMat.shape[0]]=gtMat[:,2]
            heights[objCount:objCount+gtMat.shape[0]]=gtMat[:,3]
            objCount+=gtMat.shape[0]
        widths=np.sort(widths[:objCount])[[objCount/6,2*objCount/6,3*objCount/6,4*objCount/6,5*objCount/6]]
        heights=np.sort(heights[:objCount])[[objCount/6,2*objCount/6,3*objCount/6,4*objCount/6,5*objCount/6]]
        #[go(cmd) for cmd in list(set([('mkdir -p '+('/'.join(f.split('/')[:-2]))+'/conf_rnd') for f in sys.argv[2]]))]
        for fname in sys.argv[2:]:
            ofname=fname.split('/')
            ofname[-2]='conf_rnd'
            cmd='mkdir -p '+'/'.join(ofname[:-1])
            print cmd
            go(cmd)
            ofname='/'.join(ofname)[:-3]+'csv'
            sz=(1280,720)
            res=np.random.randint(0,100000000,[100000,5],dtype='int32')
            res[:,2]=widths[res[:,2]%5]
            res[:,3]=heights[res[:,3]%5]
            res[:,0]=res[:,0]%(sz[0]-res[:,2])
            res[:,1]=res[:,1]%(sz[1]-res[:,3])
            print (sz[1]-res[:,1])
            res=res.astype('float')
            res[:,4]=np.random.rand(res.shape[0])
            res=res[np.argsort(-res[:,4]),:]
            open(ofname,'w').write('\n'.join([','.join([str(c) for c in l]) for l in res.tolist()]))
        sys.exit(0)


    if sys.argv[1]=='conf2IoU':
        createRequiredDirs(sys.argv[2:],'+../iou_')
        if switches['dontCareDictFile']!='':
            dictionary=[l for l in open(switches['dontCareDictFile']).read().split('\n') if len(l)]
        else:
            dictionary=[]
        def worker(confFname):
            t=time.time()
            confMat=fname2Array(confFname)
            #print confMat.shape
            if confMat.shape[0] == 5:
                confMat = np.array([confMat])
            #print confMat.shape
            idx=np.argsort(-confMat[:,4]) #For others (Suppression, FCN)
            #idx=np.argsort(confMat[:,4]) #For TP
            confMat=confMat[idx,:]
            #confMat=confMat[reversed(idx),:]#UGLY!!!!!!!!!!!!!!!!!!!!!!!!!!
            try:
                gtMat,transcriptions=loadTxtGtFile(getGtFromConf(confFname))
            except:
                open("/tmp/blacklist.txt","a").write(confFname.split("/")[-1] + '\n')
                lines=[[' '.join(c.split()) for c in l.split(",") ] for l in  open(getGtFromConf(confFname)).read().split() if len(l.split(","))==5]
                cleanLines=[]
                cleanTranscriptions=[]
                for line in lines:
                    try:
                        cleanLines.append([float(c) for c in line[:4]] )
                        cleanTranscriptions.append(line[-1])
                    except:
                        pass
                gtMat=np.array(cleanLines)
                transcriptions=np.empty(gtMat.shape[0],dtype=object)
                transcriptions[:]=cleanTranscriptions
            #except:
            #    print "BADGT: ",confFname
            dontCare=getDontCare(transcriptions,dictionary)
            (IoU,I,U,minsurf)=get2PointIU(gtMat,confMat)
            augmentedIoU=np.empty([IoU.shape[0]+1,IoU.shape[1]])
            augmentedIoU[:-1,:]=IoU
            augmentedIoU[-1,:]=dontCare
            #print 'Transcriptions: ',transcriptions
            #print 'DONT CARE: ',dontCare
            #array2csvFname(IoU,getIouFromConf(confFname))
            array2pngFname(augmentedIoU,getIouFromConf(confFname))
            #open(getIouFromConf(confFname),'w').write(arrayToCsvStr(IoU))
            print confFname, ' to ', getIouFromConf(confFname), ' ', int((time.time()-t)*1000)/1000.0,' sec.'
            return None
        pool=Pool(int(switches['threads']))
        if eval(switches['threads'])==1:
            [worker(f) for f in sys.argv[2:]]
        else:
            pool=Pool(int(switches['threads']))
            pool.map(worker,sys.argv[2:])
        sys.exit(0)


    if sys.argv[1]=='prop2conf':
        def worker(propFname):
            proposals=fname2Array(propFname)
            propRects=proposals[:,:5]#.astype('int32')
            confidenceDict=defaultdict(list)

            for rectId in range(proposals.shape[0]):
                rect=tuple(propRects[rectId,:])
                confidenceDict[rect].append(proposals[rectId,4])
            newProds=reversed(sorted([(max(confidenceDict[rec]),rec) for rec in confidenceDict.keys()]))

            arr=np.array([list(s[1])+list([s[0]]) for s in newProds],dtype='float')
            array2csvFname(arr,getConfidenseFromProposal(propFname))
        createRequiredDirs(sys.argv[2:],'+../conf_')
        pool=Pool(int(switches['threads']))
        if eval(switches['threads'])==1:
            [worker(f) for f in sys.argv[2:]]
        else:
            pool=Pool(int(switches['threads']))
            pool.map(worker,sys.argv[2:])
        sys.exit(0)



    if sys.argv[1]=='getCumRecall':
        def getConfFromConf(fname,confName):
            if confName=='.':
                return fname
            else:
                return '/'.join(fname.split('/')[:-2]+[confName]+fname.split('/')[-1:])
        iouThr=eval(switches['IoUThresholds'])
        maxProposals=eval(switches['maxProposalsIoU'])
        #resGtObjDetected=np.zeros([maxProposals,len(iouThr)],dtype='int64')
        resGtObjCount=0
        confDict=eval(switches['extraPlotDirs'])
        resGtObjDetected=dict([(k,np.zeros([maxProposals,len(iouThr)],dtype='int64')) for k in confDict.keys()])
        for confFname in sys.argv[2:]:
            for confStr in confDict.keys():
                #print 'Conf ',getConfFromConf(confFname,confStr)
                #print 'IoU ',getIouFromConf(getConfFromConf(confFname,confStr))
                try:
                    augmentedIoU=fname2Array(getIouFromConf(getConfFromConf(confFname,confStr)))
                except:
                    print 'Could not load ',getIouFromConf(getConfFromConf(confFname,confStr))
                    raise Exception()
                if augmentedIoU.shape[0]>1:
                    care=augmentedIoU[-1,:]
                    keepProposals=min(maxProposals,augmentedIoU.shape[0]-1)
                    if eval(switches['care']):
                        IoU=augmentedIoU[:keepProposals,care.astype('bool')]
                    else:
                        IoU=augmentedIoU[:keepProposals,:]
                    #print care
                    #print IoU.shape
                    resGtObjCount+=IoU.shape[1]
                    #dontCare=augmentedIoU[:-1,:]
                    #IoU[:,np.argmax(IoU,axis=1)]*=2#removin non maximal matches (a proposal identifies at most an object)
                    #IoU-=augmentedIoU[:keepProposals,:]
                    for tNum in range(len(iouThr)):
                        thr=iouThr[tNum]
                        found=((IoU>=thr).cumsum(axis=0)>0)
                        resGtObjDetected[confStr][:keepProposals,tNum]+=found.sum(axis=1)
                        resGtObjDetected[confStr][keepProposals:,tNum]+=found.sum(axis=1).max()
                    #print IoU[:3,:]
        sortedKeysByLegends=[e[1][0] for e in sorted([(confDict[k],(k,confDict[k])) for k in confDict.keys()])];colors="rgbcmyk"*10

        count=0
        for confStr in sortedKeysByLegends:
        #for confStr in confDict.keys():

            plt.plot(resGtObjDetected[confStr].astype('float')/(resGtObjCount/len(confDict.keys())), colors[count] ,label=confDict[confStr])
            count+=1
            #plt.legend(confDict[confStr])
        plt.ylim(0,1)
        plt.xscale('log')
        plt.xlabel('# of proposals')
        plt.ylabel('Detection Rate')
        plt.legend(loc='upper left')
        #plt.title('COCO-Text')
        #plt.title('ICDAR')
        plt.grid()
        plt.savefig(switches['plotfname'])
        sys.exit(0)


