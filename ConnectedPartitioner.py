'''
Created on 2013-03-24

@author: Alan
'''

from ConnectedPartitions import CPartitions
from DistanceMatrix import DistanceMatrix
import GraphGen
from BreadthFirstExplorer import BreadthFirstExplorer
from Graph import Graph

class CPartitioner(object):
    '''
    iterator over all partitionings of the nodes of graph such that each partition
    is a connected subgraph
    '''

    def __init__(self, graph, distanceMatrix):
        self.g = graph
        self.dm = distanceMatrix
        self.done = False
        self.recurIter = None
        if len(graph.nodes) == 0:
            self.leadIter = None
        else:
            self.root = list(graph.nodes)[0]
            self.leadIter = BreadthFirstExplorer(graph, distanceMatrix, [self.root])
            self.leadPartition = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.leadIter == None:
            raise StopIteration
        
        if self.recurIter == None:
            try:
                self.leadPartition = self.leadIter.__next__()
            except StopIteration:
                self.leadIter = None
                return self.__next__()
            if len(self.leadPartition) == len(self.g.nodes):
                return [self.leadPartition] 
            remainingNodes = self.g.nodes - set(self.leadPartition) 
            remainingGraph = self.g.getSubgraph(remainingNodes)
            self.recurIter = CPartitioner(remainingGraph, DistanceMatrix(remainingGraph))
            return self.__next__()
        else:
            try:
                restPartition = self.recurIter.__next__()
            except StopIteration:
                self.recurIter = None
                return self.__next__()
            return [self.leadPartition] + restPartition

    def print(self):
        print(self.leadPartition)
    
def testme():
    g = Graph(4)
    g.addEdgeList([(0,1),(0,2),(2,3),(1,2)])
    prt = CPartitioner(g, DistanceMatrix(g))
    #prt.print()
    for x in prt:
        print(sorted(x))

# Testing!
#testme()

