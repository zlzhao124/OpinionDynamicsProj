'''
Created on 2013-03-16

@author: Alan
'''
from Graph import Graph
from DistanceMatrix import DistanceMatrix
from reprlib import recursive_repr
from pickle import STOP
import GraphGen
from NBoolIterator import NBoolIterator
import itertools

class BreadthFirstExplorer(object):
    '''
    This class is an iterator that will iterate over all connected subgraphs of a given graph, radiating out 
    from a given root set (i.e. all subgraphs will contain all nodes in the root set).  Note that if the root set
    are not connected, then generated subgraphs will have one connected component corresponding to each set of 
    connected roots
    '''

    def __init__(self, graph, distanceMatrix, rootSet, nodeExcludeList=[]):
        if rootSet == None or len(rootSet)==0 :
            self.rootSet = []
            return
            #raise Exception("BreadthFirstExplorer cannot be initialized with rootSet = ",rootSet)
        ##print(">>",rootSet," x:",nodeExcludeList)
        self.g = graph
        self.nodelist = self.g.nodes.copy()
        self.dm = distanceMatrix
        self.rootSet = rootSet
        self.excludeList = nodeExcludeList + rootSet
        self.neighbourSet = distanceMatrix.getSetNeighbours(rootSet,excludeList=nodeExcludeList)
        self.excludeList = self.excludeList + self.neighbourSet 
        if len(self.neighbourSet) != 0:            
            self.nNiter = NBoolIterator(len(self.neighbourSet))   # iterator over all combinations of the neighbour set
            self.nNiter.__next__()       # skip iterator for all False
            self.workingNSet = self._getNextWorkingNSet()      # subset of neighbourSet we are currently working with
            self.restIter = self._getNextRestIterator()       # iterator over all recursive results
        else:
            self.workingNSet = None
            self.restIter = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.rootSet) == 0:
            raise StopIteration
        
        if self.workingNSet == None:
            # No more root sets
            temp = self.rootSet
            self.rootSet = []
            return temp
        elif self.restIter == None:
            try:
                # load next working set
                self.workingNSet = self._getNextWorkingNSet()
                self.restIter = self._getNextRestIterator()
                return self.__next__()
            except StopIteration:
                self.workingNSet = None
                return self.__next__()
        else:
            try:
                p = self.restIter.__next__()
            except StopIteration:
                self.restIter = None
                return self.__next__()
            else:
                return self.rootSet + p

    #end def
    
    def _getNextWorkingNSet(self):
        ##print("_getNextWorkingNSet: r=",self.rootSet)
        rbools = self.nNiter.__next__()
        wneighs = [r for (b,r) in zip(rbools,self.neighbourSet) if b]
        ##print("... result=",wneighs)
        return wneighs
    
    def _getNextRestIterator(self):
        return BreadthFirstExplorer(self.g, self.dm, self.workingNSet,nodeExcludeList=self.excludeList)

def _testme():
    g = GraphGen.makeBipartite(1, 4)
    g.addNode(5)
    g.addEdgeList([(5,1),(5,2),(5,3),(4,5)])
    t = BreadthFirstExplorer(g, DistanceMatrix(g),[0])
    for x in t:
        print(x)

#_testme()