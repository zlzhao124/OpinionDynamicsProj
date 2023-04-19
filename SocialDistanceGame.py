'''
Created on 2013-02-12

@author: Alan

Generates the utility for all nodes in a Graph g with DistanceMatrix dMat.  
self.sd = dictionary containing social distance game utilities (indexed on node) 
self.nodes = list of nodes

Assumes all components in g are connected
'''
from Graph import Graph
from DistanceMatrix import DistanceMatrix

class SocDistance(object):

    def __init__(self, g, dMat, ForbidDisconnectGraph=True):
        '''
        Initializes object based on Graph g and associated DistanceMatrix dMat
        '''
        self.sd = {}            # utilities
        self.nodes = g.nodes    # list of nodes
        for v in g.nodes:
            tally = 0
            nNodes = 0
            for k in g.nodes:
                if v == k : continue
                if dMat.d[v][k] == dMat.INF:
                    # node in different component
                    if ForbidDisconnectGraph:
                        raise Exception("Attempting to build social distance matrix on disconnected graph")
                    # otherwise do nothing
                else:
                    # node found in same component
                    tally = tally + 1 / dMat.d[v][k]
                    nNodes = nNodes + 1
            if nNodes == 0: # unnecessary special case
                self.sd[v] = 0
            else:
                self.sd[v] = tally / (nNodes+1)

    def update(self,that):
        '''
        Combines two SocDistance objects (copies over entries in self.sd and self.nodes)
        '''
        self.sd.update(that.sd)
        self.nodes.update(that.nodes)

    def getSocialWelfare(self):
        '''
        Returns the sum of all sd utilities for all nodes (the social welfare 
        of the subgraph, or entire graph if proper updates have been called))
        '''
        socialWelfare = 0
        for v in self.nodes:
            socialWelfare = socialWelfare + self.sd[v]
        return socialWelfare

    def print(self):
        print(self.toString())
    
    def toString(self):
        st = ""
        for v in self.nodes:
            st = st + "({0}:{1}) ".format(v,self.sd[v])
        return st
    
    def copy(self):
        sd_new = SocDistance(Graph(), DistanceMatrix(Graph()))
        sd_new.sd = self.sd.copy()
        sd_new.nodes = self.nodes.copy()
        return sd_new
    
def _testme():
    g1 = Graph()
    g1.addNode(0)
    g1.addNode(1)
    g1.addNode(2)
    g1.addNode(3)
    g1.addEdge(0,1)
    g1.addEdge(1,2)
    g1.addEdge(0,2)
    g1.addEdge(2,3)
    dmat = DistanceMatrix(g1)
    sdmat = SocDistance(g1,dmat)
    #sdmat.print()
    g1.addEdge(0,3)
    g1.addEdge(1,3)
    dmat = DistanceMatrix(g1)
    sdmat = SocDistance(g1,dmat)
    #sdmat.print()
    g2 = Graph(4)
    g2.addEdge(0,1)
    g2.addEdge(1,2)
    g2.addEdge(2,3)
    dmat = DistanceMatrix(g2)
    sdmat = SocDistance(g2,dmat)
    #sdmat.print()
    g3 = Graph(1)
    dmat = DistanceMatrix(g3)
    sdmat = SocDistance(g3,dmat)
    sdmat.print()

#_testme()