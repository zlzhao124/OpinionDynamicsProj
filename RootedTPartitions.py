'''
Created on 2013-02-12

@author: Alan
'''
from Graph import Graph
from ShortestPaths import DistanceMatrix
from reprlib import recursive_repr
from pickle import STOP
import GraphGen
from NBoolIterator import NBoolIterator
import itertools

class RootedTPartitions(object):
    '''
    classdocs
    THIS DOESN"T WORK YO.
    '''

    def __init__(self, graph, distanceMatrix, root=None):
        self.g = graph
        self.nodelist = self.g.nodes.copy()
        self.dm = distanceMatrix
        if root == None:
            if(len(self.nodelist)==0):
                self.root = None
            else:
                self.root = self.nodelist.pop()
        else:
            self.root = root
        
        if root == None:
            pass
        else:
            self.neighbours = self._getNeighbours(self.root)
            self.cPartitions = self._getChildPartitions(self.root)
            self.cpNiter = NBoolIterator(len(self.cPartitions))
            self.cpNiter.__next__()     # skip "all False"
            self.currIterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        #print("n>",self.neighbours)
        #print("r>",self.remainingNodes)
        if self.root == None:
            raise StopIteration
        
       
        if self.currIterator != None:
            try:
                p = self.currIterator.__next__()
            except StopIteration:
                self.currIterator = None
                return self.__next__()
            else:
                print(">>",self.root,"|",p)
                return [self.root].append(p)
        else:
            try:
                # load next multiterator
                self.currIterator = self._getNextMultiterator()
                return self.__next__()
            except StopIteration:
                # out of multiterators, gig's up, return [root] and be done
                temp = self.root
                self.root = None
                return [temp]
    #end def
    
    def _getNeighbours(self,v):
        return [x for x in self.g.nodes if self.dm.d[v][x] == 1]

#    def _getNeighbourRepresentatives(self,v):
#        '''
#        Must be run before other branches of __next__() which will start modifying self.neighbour and self.remainingNodes
#        '''
#        sg = self.g.getSubgraph(self.remainingNodes)
#        dm = DistanceMatrix(sg)
#        reps = self.neighbours.copy()
#        for x in reps:
#            for y in reps:
#                if x!=y and dm.d[x][y] != dm.INF:
#                    reps.remove(y)
#        return reps
    
    def _getChildPartitions(self,v):
        '''
        Returns the nodes of G-v, partitioned by connected components of G-v.  This is organized as a tuple, with each 
        element being a list of nodes.  The first node of each list is a neighbour of v. 
        '''
        remainingNodes = [x for x in self.nodelist if x != v]
        sg = self.g.getSubgraph(remainingNodes)
        dm = DistanceMatrix(sg)
        #forbiddenNeighbours = []
        childPartitions = []
        for x in self.neighbours:
            #if x in forbiddenNeighbours: continue
            x_reachable = [x]
            for y in remainingNodes:     #OPT: to remove y from the list when it's been added to x_reachable
                if y==x: continue
                if dm.d[x][y] != dm.INF:
                    x_reachable.append(y)
                    #if y in self.neighbours:
                    #    forbiddenNeighbours.append(y)
            childPartitions.append(x_reachable)
        return childPartitions
    
    def _getPartitionIterator(self,cPartition):
        sg = self.g.getSubgraph(cPartition)
        return RootedTPartitions(sg, DistanceMatrix(sg), cPartition[0])
    
    def _getNextMultiterator(self):
        bitList = self.cpNiter.__next__()
        iterList = [self._getPartitionIterator(i) for (b,i) in zip(bitList,self.cPartitions) if b]
        return itertools.product(*iterList)     #OPT: get rid of itertools

#def _listdiff(a,b):
#    return list(set(a)-set(b))    

def _testme():
    g = Graph(3)
    g.addEdgeList([(0,1),(1,2),(0,2)])
    t = RootedTPartitions(g, DistanceMatrix(g),1)
    for x in t:
        print(x)

_testme()