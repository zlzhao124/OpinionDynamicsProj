'''
Created on 2013-03-17

@author: Alan
'''

from BreadthFirstExplorer import BreadthFirstExplorer
from Graph import Graph
from DistanceMatrix import DistanceMatrix
import GraphGen

class CPartitions(object):
    '''
    classdocs
    iterator over all lists of nodes in graph that represent connected subgraphs
    '''

    def __init__(self, graph, distanceMatrix):
        self.g = graph
        self.dm = distanceMatrix
        self.itemlist = graph.nodes.copy()
        self.iter = None        # BFE object to iterate over
        self.currentRoot = None
        self.excludeList = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == None:
            if len(self.itemlist) == 0:
                raise StopIteration
            if self.currentRoot != None:
                self.excludeList.append(self.currentRoot)
            self.currentRoot = self.itemlist.pop()
            self.iter = BreadthFirstExplorer(self.g, self.dm, [self.currentRoot], self.excludeList)
            return self.__next__()
        else:
            try:
                p = self.iter.__next__()
            except StopIteration:
                self.iter = None
                return self.__next__()
            return sorted(p)

def _testme():
    g = Graph(5)
    g.addEdgeList([(0,1),(0,2),(0,3),(1,2),(2,4)])
    p = CPartitions(g, DistanceMatrix(g))
    for x in sorted(p):
        print(x)

#_testme()