'''
Created on 2013-02-11

@author: Alan

Produces a distace matrix based on a given Graph.  
Matrix accessible as a two-level dictionary
Uses the Floyd-Warshall algorithm
'''

from Graph import Graph

class DistanceMatrix(object):
    '''
    classdocs
    '''


    def __init__(self, gph):
        self.INF = gph.getNumNodes()
        self.nodes = gph.nodes
        self.d = {}
        # Floyd-Warshall algorithm
        for x in gph.nodes:
            self.d[x] = {}
            for y in gph.nodes:
                self.d[x][y] = self.INF
        for v in gph.nodes:
            self.d[v][v] = 0
        for i in gph.nodes:
            for j in gph.nodes:
                if (i,j) in gph.edges or (j,i) in gph.edges:
                    self.d[i][j] = 1
        for k in gph.nodes:
            for i in gph.nodes:
                for j in gph.nodes:
                    if self.d[i][k] + self.d[k][j] < self.d[i][j]:
                        self.d[i][j] = self.d[i][k] + self.d[k][j]
        
    def dist(self,i,j):
        return self.d[i][j]
    
    def isConnected(self):
        for i in self.nodes:
            for j in self.nodes:
                if self.d[i][j] == self.INF:
                    return False
        return True
    
    def getNeighbours(self,v):
        neigh = []
        for x in self.nodes:
            if self.d[v][x] == 1:
                neigh.append(x)
        return neigh
    
    def getSetNeighbours(self,vertexSet,excludeList=[]):
        '''
        Given a set S of vertices, returns all nodes adjacent to at least one vertex in S
        (excluding any member of S)
        '''
        neigh = []
        for x in self.nodes:
            if x in vertexSet or x in excludeList:
                continue
            for v in vertexSet:
                if self.d[x][v] == 1:
                    neigh.append(x)
                    break
        return neigh
    
    def getDiameter(self):
        maxDist = 0
        for i in self.nodes:
            for j in self.nodes:
                if self.d[i][j] == self.INF:
                    pass
                elif self.d[i][j] > maxDist:
                    maxDist = self.d[i][j]
        return maxDist 
    
    def print(self):
        for i in self.nodes:
            for j in self.nodes:
                if self.d[i][j] == self.INF:
                    print('-',end='')
                else:
                    print(self.d[i][j],end='')
            print("")
        print("")
    
def testme():
    g = Graph(6)
    g.addEdge(0,1)
    g.addEdge(1,2)
    g.addEdge(0,2)
    g.addEdge(2,3)
    g.addEdge(5,4)
    dmat = DistanceMatrix(g)
    dmat.print()
    print("Diameter: ",dmat.getDiameter())
        
#testme()
