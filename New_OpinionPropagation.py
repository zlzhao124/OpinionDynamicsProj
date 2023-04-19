'''
Created on 2013-03-31

Runs the "Skeptical Bayes" opinion dynamics simulation.

Parameters:

-o	output file prefix (experiment name); defaults to 'x'
	output files are placed into a directory called opinion_prop, and are suffixed with "_data.txt" and "_devi.txt"

# graph specifications
-graph		name of graph algorithm
-n			number of vertices
-mlist		a list of attachment parameters for the graph model (default=[1,2,3,4,5,6])

# propagation specifications
-trust		trust model being used: [uniform|degree|kernel]
-initial	initial distribution of opinions [u|2t|b] (default: u)
-onefrac	initial fraction of 1-extremists (default: 0.1)
-zerofrac	initial fraction of 0-extremists (default: 0.1)
-maxIter	maximum # of iterations per simulation before hard stop (default: 500)
-delta		delta parameter (default: 0.001)
-fickle		fickleness parameter (default: 1.5)
-mutation	probability of random changes in opinion (default: 0.0)
-trials		number of trials to conduct per experimental condition (default: 1)
-elist		a list of empathy parameters for the dynamics process
			(default=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
-conviction	conviction parameter (default: 0.0)
    

Example usage:
python OpinionPropagation.py -o expr1 -graph=BA -n 200 -trust=uniform -trials=25 -zerofrac=0 > expr1_console.txt

@author: Alan
'''

from Graph import Graph, graphEval
from VertexWeights import VertexWeights
import GraphGen
from Graph import Graph
import random
import math
import argparse
import sys
from argparse import ArgumentError

DATA_LOGFILE = ""
SUMMARY_LOGFILE = ""
TRACK_LOGFILE = ""
DEVI_LOGFILE = ""
TRACK = -1

''' ===========================
    EXPERIMENT PARAMETERS 
=========================== '''

graphType = -1
nVertices = -1
oneFraction = -1 
zeroFraction = -1
maxIterations = -1
deltaStop = -1         # stop when maxDelta < deltaStop
FICKLENESS = -1
mutationChance = -1
TRIALS_PER = -1
trustModel = -1
initialOpinions = -1
conviction = -1

def indoctrinateVertices(opinions, vertices, val=1):
    '''
    Forces vertices to have opinion val 
    '''
    for v in vertices:
        opinions.setWeight(v, val)
        
def randomizeOpinions(opinions,distr=None):
    if distr==None or distr=="uniform":
        for v in opinions.getIterator():
            opinions.setWeight(v,random.random())
    elif distr=="2tri":
        for v in opinions.getIterator():
            if random.random()<0.5:
                r = random.triangular(0,0.5,0.25)
            else:
                r = random.triangular(0.5,0.75,1)
            opinions.setWeight(v,r)
    elif distr=="beta":
        for v in opinions.getIterator():
            opinions.setWeight(v,random.betavariate(0.2,0.2))

def mutateOpinions(opinions,chance,excludeVertices):
    for v in opinions.getIterator():
        if v in excludeVertices:
            pass
        elif random.random() <= chance:
            newOp = opinions.getWeight(v) + random.gauss(0,0.15)
            if newOp<0: newOp=0
            if newOp>1: newOp=1
            #print("Vertex {0} mutate {1} --> {2}".format(v,opinions.getWeight(v),newOp))
            opinions.setWeight(v,newOp)

def getMeanOpinion(opinions,excludeSet=[]):
    tally = 0
    denom = 0
    for v in opinions.getIterator():
        if v in excludeSet:
            pass
        else:
            tally += opinions.getWeight(v)
            denom += 1 
    return tally / denom

def getMeanStdOpinion(opinions,excludeSet=[]):
    (m,v) = getMeanVarianceOpinion(opinions,excludeSet)
    return (m,math.sqrt(v))

def getMeanVarianceOpinion(opinions,excludeSet=[]):
    mean = getMeanOpinion(opinions,excludeSet)
    tally = 0
    denom = 0
    for v in opinions.getIterator():
        if v in excludeSet:
            pass
        else:
            v_var = abs(opinions.getWeight(v) - mean)
            tally += v_var ** 2
            denom += 1
    variance = tally / denom
    return (mean,math.sqrt(variance))

def getMeanSkewnessOpinion(opinions,excludeSet=[]):
    (mean,std) = getMeanStdOpinion(opinions,excludeSet)
    tally = 0
    denom = 0
    for v in opinions.getIterator():
        if v in excludeSet:
            pass
        else:
            v_var = abs(opinions.getWeight(v) - mean)
            tally += v_var ** 3
            denom += 1
    if std == 0:
        skewness = 0
    else:
        skewness = (tally / denom) / (std**3)

#    if denom == 0 or std == 0:
#        print("Tally = " + str(tally) + ", denom = " + str(denom) + ", std = " + str(std))

    #skewness = (tally / denom) / (std**3)
    return (mean,skewness)


def getMeanKurtosisOpinion(opinions,excludeSet=[]):
    (mean,variance) = getMeanVarianceOpinion(opinions,excludeSet)
    tally = 0
    denom = 0
    for v in opinions.getIterator():
        if v in excludeSet:
            pass
        else:
            v_var = abs(opinions.getWeight(v) - mean)
            tally += v_var ** 4
            denom += 1
    if variance == 0:
        skewness = 0
    else:
        skewness = (tally / denom) / (variance**2)
    return (mean,skewness)

def getAvgDistFromHalf(opinions,excludeSet=[]):
    tally = 0
    denom = 0
    for v in opinions.getIterator():
        if v in excludeSet:
            pass
        else:
            tally += abs(opinions.getWeight(v) - 0.5)
            denom += 1
    return tally / denom

def getAvgSquaredDistFromHalf(opinions,excludeSet=[]):
    tally = 0
    denom = 0
    for v in opinions.getIterator():
        if v in excludeSet:
            pass
        else:
            tally += abs(opinions.getWeight(v) - 0.5)**2
            denom += 1
    return tally / denom

def getMaxDelta(graph, oldOpinions, newOpinions):
    maxDelta = 0
    for v in graph.nodes:
        delta = abs(newOpinions.getWeight(v) - oldOpinions.getWeight(v))
        if delta > maxDelta:
            maxDelta = delta
    return maxDelta

def diffOpinions(x,y,opinions):
    return abs(opinions.getWeight(x)-opinions.getWeight(y))


def diffuseByAverage(graph, opinions, threshold=None, getEdgeWeight=None, weightSelf=True):
    '''
    One iteration of opinion propagation, using averaging model
        threshold = -1 if continunous
        threhsold = t if opinion <- 0 if avg < t; 1 otherwise
        getEdgeWeight = function (v,x,opinions) --> trust weight of v toward x
    '''
    newOp = VertexWeights(len(graph.nodes))
    for v in graph.nodes:
        n_v = graph.getNeighbours(v)
        deg = graph.getNodeDegree(v)
        #if i don't have neighbors, my opinions don't change
        if deg == 0:
            newOp.setWeight(v, opinions.getWeight(v))
            continue
            
        #assuming my own weight is my degree, initialize tally and denom to the first term in Equation 2
        if weightSelf:
            tally = opinions.getWeight(v) * deg * (1-conviction)
            denom = deg * (1-conviction)
        else:
            tally = 0
            denom = 0
        ##print("v=",v)
        
        #summing up over my neighbors to get second term in Equation 2
        for x in n_v:
            if getEdgeWeight == None:
                tally += opinions.getWeight(x)
                denom += 1
            else:
                tally += opinions.getWeight(x) * getEdgeWeight(v,x,opinions)
                denom += getEdgeWeight(v,x,opinions)
                ##print(" neighbour x=",x," opinion=",opinions.getWeight(x)," trust=",getEdgeWeight(v,x,opinions))
        
        
        if denom == 0:
            print("Tally = " + str(tally) + ", denom = " + str(denom))

        #computes Equation 2
        tally = tally / denom
        if threshold==None:
            if tally < 0:
                tally = 0
            elif tally > 1:
                tally = 1
        else:
            if tally < threshold(v):
                tally = 0
            else:
                tally = 1
        newOp.setWeight(v, tally)
        ##print(" opinion[",v,"] <-- ",tally)
    return newOp

''' ===========================
    TRUST WEIGHT FUNCTIONS 
=========================== '''

def linearEdgeWeight(v,neighbour_v,opinions):
    delta = abs(opinions.getWeight(v)-opinions.getWeight(neighbour_v))
    # delta should be in [0,1]
    return 1-delta

STEP_LOW = 0.25
STEP_HIGH = 0.75
def stepEdgeWeight(v,neighbour_v,opinions):
    delta = diffOpinions(v,neighbour_v,opinions)
    # delta should be in [0,1]
    if delta <= STEP_LOW:
        return 1
    elif delta <= STEP_HIGH:
        return 0.5
    else:
        return 0

######### THESE ARE THE CHANGING EQUATIONS ##########
######### Both equations 2 and 3 change #############
#####################################################
#####################################################

## This is Equation 1 in the paper

##### THIS IS THE ORIGINAL EQUATION ########
#def getGaussianEdgeWeight(v, neighbour_v, opinions, bandwidth):
#    return math.exp(- (diffOpinions(v,neighbour_v,opinions)**2) / bandwidth) - conviction
############################################

def getGaussianEdgeWeight(v, neighbour_v, opinions, d1, d2):

    x = diffOpinions(v,neighbour_v,opinions)

    #To the left side of d1 - pull
    if abs(x) < d1:
        weight = math.exp(- ((diffOpinions(v,neighbour_v,opinions) - d1)**2) / (-(((d1)**2)/math.log(2)))) - 1

    #Between d1 and d2 - neutral
    elif abs(x) < d2:
        weight = 0
        
    #To the right side of d2 - push
    else:
    #elif abs(x) > d2:
        weight = 1 - math.exp(((x - d2)**2)/(((1 - d2)**2)/math.log(2)))


    return weight

## This is equation 3 in the paper

####def getUpdatedDriftingTrustMatrix(trustMatrix,empathy,fickleness,opinions):
####    newTrust = uniformInitialTrust(1)
####    for x in range(nVertices):
####        for y in range(nVertices):
####            newTrust[x][y] = (trustMatrix[x][y] + fickleness*getGaussianEdgeWeight(x, y, opinions, empathy))/(1+fickleness)
####    return newTrust



def getUpdatedDriftingTrustMatrix(trustMatrix,empathy,fickleness,opinions, alpha, d1, d2):
    newTrust = uniformInitialTrust(1)
    for x in range(nVertices):
        for y in range(nVertices):
            newTrust[x][y] = (alpha * trustMatrix[x][y]) + ((1 - alpha) * (getGaussianEdgeWeight(x, y, opinions, d1, d2)))
    return newTrust



######################################################
######################################################
######################################################
######################################################

def dynamicEdgeWeight(trustMatrix):
    return lambda v, neighbour_v, opinions: trustMatrix[v][neighbour_v]

def getUpdatedNAIVETETrustMatrix(trustMatrix,naivete,opinions):
    newTrust = uniformInitialTrust(1)
    for x in range(nVertices):
        for y in range(nVertices):
            newTrust[x][y] = trustMatrix[x][y] / (naivete ** diffOpinions(x, y, opinions))
    return newTrust
def uniformInitialTrust(val):
    return [[val]*nVertices]*nVertices
def degreeBasedTrust(graph):
    trustMat = uniformInitialTrust(0)
    for v in graph.nodes:
        for x in graph.nodes:
            trustMat[v][x]=graph.getNodeDegree(x)/graph.getNodeDegree(v)
    return trustMat
def kernelTrust(graph,opinions,bandwidth):
    trustMat = uniformInitialTrust(0)
    for v in graph.nodes:
        for x in graph.nodes:
            trustMat[v][x]=getGaussianEdgeWeight(v, x, opinions, bandwidth)
    return trustMat

''' ===========================
    THRESHOLD FUNCTION 
=========================== '''

def flatThreshold(v):
    return 0.5

''' ===========================
    DETAILED LOGGING 
=========================== '''
def frange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def logOpinions(opinions,excludeList,LOGFILE):
    NUM_INTERVALS = 20
    buckets = [0]*NUM_INTERVALS
    for v in range(nVertices):
        if v in excludeList:
            continue
        b = math.floor(opinions.getWeight(v) * NUM_INTERVALS)
        if b == NUM_INTERVALS: b=NUM_INTERVALS-1
        buckets[b] += 1
    for i in range(NUM_INTERVALS):
        LOGFILE.write("{0}  ".format(buckets[i]))
    LOGFILE.write("\n")
        

''' ===========================
        MAIN PROGRAM 
=========================== '''

def runExperiment(mAttachmentFactor,empathy,fickleness, d1, d2, alpha):
        
    #initialize opinions/traits
    opinions = VertexWeights(nVertices, 0)
    randomizeOpinions(opinions,distr=initialOpinions)
    
    #initialize opinions/traits of extremists
    numExtreme1 = int(nVertices * oneFraction)
    numExtreme0 = int(nVertices * zeroFraction)
    extremists = random.sample(list(range(0,nVertices)),numExtreme1+numExtreme0)
    onextremists = extremists[:numExtreme1]
    onextremists = sorted(onextremists)
    zeroxtremists = extremists[numExtreme1:]
    zeroxtremists = sorted(zeroxtremists)
    if len(zeroxtremists) != numExtreme0: raise Exception("len(zeroxtremists) != numExtreme0") 
    indoctrinateVertices(opinions,onextremists,1)
    indoctrinateVertices(opinions,zeroxtremists,0)
    #moderateVertices = sorted([v for v in g.nodes if v not in indoctrinatedVertices])
    
    if graphType=="BA":
        g = GraphGen.makeRandomBAGraph(nVertices, GraphGen.makeClique(2), mAttachmentFactor)
    elif graphType=="ER":
        g = GraphGen.makeRandomGraph(nVertices, mAttachmentFactor, ConnectedOnly=True)
    elif graphType=="hBA":
        g = GraphGen.makeHomophilyBAGraph(nVertices, GraphGen.makeClique(2), mAttachmentFactor, opinions, empathy, conviction)
    elif graphType=="hER":
        g = GraphGen.makeHomophilyERGraph(nVertices, mAttachmentFactor, opinions, ConnectedOnly=True)
    else:
        raise argparse.ArgumentTypeError("Unexpected graph type: {0}".format(graphType))
    #gstring = "{(64, 0), (78, 48), (79, 55), (63, 27), (21, 6), (77, 52), (59, 41), (8, 5), (38, 37), (4, 0), (45, 22), (6, 3), (40, 13), (94, 53), (36, 8), (16, 3), (17, 2), (74, 56), (13, 7), (99, 3), (18, 9), (94, 20), (48, 21), (97, 0), (68, 65), (98, 7), (25, 10), (3, 2), (88, 4), (11, 0), (90, 68), (58, 48), (99, 58), (0, 1), (21, 15), (45, 2), (69, 5), (39, 27), (70, 10), (23, 9), (30, 11), (33, 9), (56, 6), (84, 4), (41, 6), (12, 8), (74, 38), (89, 49), (43, 8), (49, 0), (46, 7), (84, 55), (2, 1), (47, 6), (79, 72), (9, 4), (75, 21), (57, 2), (29, 4), (60, 25), (59, 3), (35, 3), (12, 2), (11, 10), (15, 0), (20, 0), (85, 62), (25, 0), (22, 7), (65, 43), (33, 2), (6, 4), (62, 58), (10, 4), (80, 42), (31, 4), (54, 49), (64, 4), (67, 11), (14, 6), (91, 11), (96, 9), (56, 10), (71, 2), (93, 29), (43, 39), (32, 0), (81, 2), (55, 29), (71, 24), (85, 8), (16, 1), (57, 17), (83, 54), (63, 19), (14, 3), (89, 53), (86, 49), (91, 0), (70, 4), (44, 15), (49, 9), (3, 0), (69, 17), (9, 8), (76, 2), (81, 1), (54, 5), (83, 0), (75, 26), (24, 9), (39, 3), (27, 20), (26, 20), (46, 4), (28, 23), (90, 26), (93, 10), (61, 33), (52, 7), (96, 55), (55, 26), (29, 3), (77, 9), (34, 5), (72, 58), (37, 8), (42, 8), (37, 5), (95, 75), (68, 8), (17, 10), (51, 36), (48, 3), (92, 8), (26, 4), (8, 7), (97, 93), (62, 12), (53, 12), (31, 10), (5, 3), (7, 0), (86, 4), (60, 10), (41, 3), (98, 43), (13, 5), (19, 5), (24, 3), (73, 11), (22, 1), (23, 4), (28, 4), (82, 2), (10, 6), (34, 11), (95, 6), (87, 20), (36, 21), (15, 8), (82, 28), (20, 8), (19, 0), (18, 8), (42, 29), (65, 62), (72, 26), (44, 8), (51, 8), (7, 4), (50, 15), (30, 9), (53, 2), (67, 53), (40, 9), (87, 46), (61, 23), (66, 4), (76, 47), (92, 4), (73, 1), (78, 73), (50, 0), (66, 2), (47, 0), (80, 21), (35, 6), (58, 7), (88, 0), (38, 4), (2, 0), (32, 5), (4, 3), (27, 8), (5, 2), (52, 48)}"
    #g = Graph.graphEval(gstring, nVertices)
    print("Graph created")
    
    if trustModel=="uniform":
        trustMatrix = uniformInitialTrust(1.0)
    elif trustModel=="degree":
        trustMatrix = degreeBasedTrust(g)
    elif trustModel=="kernel":
        trustMatrix = kernelTrust(g, opinions, empathy)
    else:
        raise argparse.ArgumentTypeError("Unexpected trust model type: -trust={0}".format(trustModel))

    
    print("Graph")
    print(g.toEval())
    print("1-Indoctrinated vertices: ", onextremists)
    print("0-Indoctrinated vertices: ", zeroxtremists)
    print("Pajek")
    print(g.outputToPajek())
    
    (m_i,s_i) = getMeanStdOpinion(opinions,extremists)
    print("Initial Opinions:: mean=",m_i," std=",s_i)
    
    for i in range(maxIterations):
        #updating weights (Equation 3 in paper)
        trustMatrix = getUpdatedDriftingTrustMatrix(trustMatrix, empathy, fickleness, opinions, alpha, d1, d2) ##
        logOpinions(opinions, extremists, TRACK)
        
        #updating opinions (Equation 2 in paper)
        newOpinions = diffuseByAverage(g,opinions,getEdgeWeight=dynamicEdgeWeight(trustMatrix))
        mutateOpinions(opinions, mutationChance, extremists)
        
        #resetting extremists' opinions to 1 or 0
        indoctrinateVertices(newOpinions,onextremists,1)
        indoctrinateVertices(newOpinions,zeroxtremists,0)
        
        mdelta = getMaxDelta(g, opinions, newOpinions)
        if mdelta < deltaStop:
            opinions = newOpinions
            break
        else:
            opinions = newOpinions
        if ((i+1) % 10 == 0):
            (m,s) = getMeanStdOpinion(opinions,extremists)
            print("iter={0}:: mean={1:.4f} std={2:.4f}".format(i+1,m,s))
            #print(opinions.outputToPajek())
    
    print("Opinions (t={0})".format(i))
    print(opinions.outputToPajek())
    (m,s) = getMeanStdOpinion(opinions,extremists)
    print("Initial Opinions:: mean={0:.4f} std={1:.4f}".format(m_i,s_i))
    print("Final Opinions  :: mean={0:.4f} std={1:.4f}".format(m,s))
    #skew = 0
    #kurt = 0
    (dummy,skew) = getMeanSkewnessOpinion(opinions,extremists)
    (dummy,kurt) = getMeanKurtosisOpinion(opinions,extremists)
    avgD = getAvgDistFromHalf(opinions, extremists)
    avgSD = getAvgSquaredDistFromHalf(opinions, extremists)
    print("                :: skew={0:.4f} kurt={1:.4f}".format(skew,kurt))
    print("                :: avgD={0:.4f} avgSD={1:.4f}".format(avgD,avgSD))
    print("kurt - skew^2 = {0:.4f} (>= 1)".format(kurt-skew**2))
    if i+1 == maxIterations:
        print("All ",i+1," iterations performed")
    else:
        print("Cascade terminated at iteration {0} (number is 1-indexed).  Max Delta={1:.4f}".format(i+1,mdelta))
    return (m,s**2,skew,kurt,i+1,avgD,avgSD)

def main():
    parser = argparse.ArgumentParser(description='Opinion propagation simulation.')
    parser.add_argument('-o', nargs='?', default='x' )
    # graph specifications
    parser.add_argument('-n', nargs=1, type=int, required=True )
    parser.add_argument('-graph', nargs=1, type=str, required=True )
    #parser.add_argument('-BA', nargs=1, type=int )
    #parser.add_argument('-ER', nargs=1, type=float )
    # propagation specifications
    parser.add_argument('-trust', nargs=1, type=str, required=True )
    parser.add_argument('-initial', nargs=1, type=str, default=["u"] )
    parser.add_argument('-onefrac', nargs=1, type=float, default = [0.1] )
    parser.add_argument('-zerofrac', nargs=1, type=float, default = [0.1] )
    parser.add_argument('-maxIter', nargs=1, type=int, default = 500 )
    parser.add_argument('-delta', nargs=1, type=float, default = 0.001 )
    parser.add_argument('-fickle', nargs=1, type=float, default = 1.5 )
    parser.add_argument('-mutation', nargs=1, type=float, default = [0.0] )
    parser.add_argument('-trials', nargs=1, type=int, default = [1] )
    parser.add_argument('-mlist', nargs=1, type=str, default=["1,2,3,4,5,6"])
    parser.add_argument('-elist', nargs=1, type=str, default=["0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1"])
    parser.add_argument('-d1list', nargs=1, type=str, default=["0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"])
    parser.add_argument('-conviction', nargs=1, type=float, default = [0.0] )
    
    args = parser.parse_args()

    global DATA_LOGFILE
    global SUMMARY_LOGFILE
    global TRACK_LOGFILE
    global TRACK
    global graphType
    global nVertices
    global trustModel
    global oneFraction 
    global zeroFraction
    global maxIterations
    global deltaStop  
    global FICKLENESS 
    global mutationChance 
    global TRIALS_PER
    global initialOpinions
    global conviction
    
    if args.graph[0]=="BA":
        graphType = "BA"
    elif args.graph[0]=="ER":
        graphType = "ER"
    elif args.graph[0]=="hBA":
        graphType = "hBA"
    elif args.graph[0]=="hER":
        graphType = "hER"
    else:
        raise argparse.ArgumentTypeError("Unexpected graph type -graph={0}".format(args.graph))
    
    if args.trust[0]=="uniform" or args.trust[0]=="u" :
        trustModel = "uniform"
    elif args.trust[0]=="degree" or args.trust[0]=="d" or args.trust[0]=="deg" :
        trustModel = "degree"
    elif args.trust[0]=="kernel" or args.trust[0]=="k" :
        trustModel = "kernel"
    else:
        raise argparse.ArgumentTypeError("Unexpected trust model: -trust={0}".format(args.trust))
    
    if args.initial[0]=="uniform" or args.initial[0]=="u" :
        initialOpinions = "uniform"
    elif args.initial[0]=="2tri" or args.initial[0]=="2t":
        initialOpinions = "2tri"
    elif args.initial[0]=="beta" or args.initial[0]=="b":
        initialOpinions = "beta"
    else:
        raise argparse.ArgumentTypeError("Unexpected initial opinion model: -initial={0}".format(args.initial))

    nVertices = int(args.n[0])
    oneFraction = float(args.onefrac[0])
    zeroFraction = float(args.zerofrac[0])
    maxIterations = args.maxIter
    deltaStop = args.delta
    FICKLENESS = args.fickle
    mutationChance = float(args.mutation[0])
    TRIALS_PER = int(args.trials[0])
    conviction = float(args.conviction[0])
    if graphType == "BA" or graphType=="hBA":
        mlist = list(map(int, args.mlist[0].split(",")))
    elif graphType == "ER" or graphType=="hER":
        mlist = list(map(float, args.mlist[0].split(",")))
    else:
        mlist = []
    elist = list(map(float, args.elist[0].split(",")))
    d1list = list(map(float, args.d1list[0].split(",")))

    DATA_LOGFILE = "opinion_prop/{0}_data.txt".format(args.o)
    DEVI_LOGFILE = "opinion_prop/{0}_devi.txt".format(args.o)
    SUMMARY_LOGFILE = "opinion_prop/{0}_summary.txt".format(args.o)
    TRACK_LOGFILE = "opinion_prop/{0}_track.txt".format(args.o)
    TRACK = open(TRACK_LOGFILE,"w")
    DATA = open(DATA_LOGFILE,"w")
    DEVI = open(DEVI_LOGFILE,"w")
    SUMM = open(SUMMARY_LOGFILE,"w")
    
    print("Global settings:")
    print("Graph: ",graphType)
    print("nVertices=",nVertices)
    print("1-extreme % =",oneFraction)
    print("0-extreme % =",zeroFraction)
    print("Fickleness =",FICKLENESS)
    SUMM.write("nVertices\t1-ex %\t0-ex %\tm\tempahty\tfickleness\tmean\tvariance\tskewness\tkurtosis\tavg D to 0.5\tavg sq D to 0.5\tnIters\n")
    
    empathy = 0.05
    
    for m in mlist:
#        for empathy in elist:
        for d1 in d1list:
            meanlist = []
            avgdlist = []
            for i in range(TRIALS_PER):
                TRACK.write("m={0},empathy={1},trial={2}\n".format(m,empathy,i+1))
                print("Setting m={0}, empathy={1}, trial={2} of {3}".format(m,empathy,i+1,TRIALS_PER))
                
                (mean,vari,skew,kurt,numIterations,avgD,avgSD)=runExperiment(m, empathy,FICKLENESS,d1, d2=1, alpha=0.75)
                
                SUMM.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(
                   nVertices,oneFraction,zeroFraction,m,empathy,FICKLENESS,mean,vari,skew,kurt,avgD,avgSD,numIterations))
                
                SUMM.flush()
                TRACK.flush()
                meanlist.append(mean)
                avgdlist.append(avgD)
            DATA.write("{0}\t".format(sum(meanlist)/TRIALS_PER))
            DATA.flush()
            DEVI.write("{0}\t".format(sum(avgdlist)/TRIALS_PER))
            DEVI.flush()
        DATA.write("\n")
        DEVI.write("\n")
    SUMM.close()
    DATA.close()
    DEVI.close

main()

''' ===========================
        ### TESTS ### 
=========================== '''

def test():
    g = Graph(5)
    g.addEdgeList([(0,1),(0,2),(0,3),(0,4),(3,4)])
    opinions = VertexWeights(5, 0.5)
    indocSet = [0]
    indoctrinateVertices(opinions, indocSet, 1)
    
    (m_i,s_i) = getMeanStdOpinion(opinions)
    for i in range(5):
        newOpinions = diffuseByAverage(g,opinions,getEdgeWeight=linearEdgeWeight)
        indoctrinateVertices(newOpinions, indocSet, 1)
        opinions = newOpinions
        print(opinions.outputToPajek())
    
    (m,s) = getMeanStdOpinion(opinions)
    print("Initial Opinions:: mean={0:.4f} std={1:.4f}".format(m_i,s_i))
    print("Final Opinions  :: mean={0:.4f} std={1:.4f}".format(m,s))
    

