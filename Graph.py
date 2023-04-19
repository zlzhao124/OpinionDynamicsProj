'''
Created on 2013-02-11

@author: Alan

Graph models a simple, nondirected graph with nodes with arbitrary labels 
supplied by the user
'''

class Graph(object):

    def __init__(self, nVertices=0):
        self.nodes = set()
        self.edges = set()
        for i in range(nVertices):
            self.addNode(i)
        
    def addNode(self, nodeID):
        if nodeID in self.nodes:
            raise Exception("NodeID {0} already in use".format(nodeID))
        self.nodes.add(nodeID)
        
    def addEdge(self, nodeA, nodeB):
        if nodeA == nodeB:
            raise Exception("Error adding edge ({0},{1}):  Self edges forbidden".format(nodeA,nodeB))
        if nodeA not in self.nodes:
            raise Exception("Error adding edge ({0},{1}):  Node {0} not in graph".format(nodeA,nodeB))
        if nodeB not in self.nodes:
            raise Exception("Error adding edge ({0},{1}):  Node {1} not in graph".format(nodeA,nodeB))
        if (nodeA,nodeB) in self.edges:
            raise Exception("Error adding edge ({0},{1}):  Edge already exists".format(nodeA,nodeB))
        if (nodeB,nodeA) in self.edges:
            raise Exception("Error adding edge ({0},{1}):  Reversed edge already exists".format(nodeA,nodeB))
        self.edges.add((nodeA,nodeB))
        
    def addEdgeList(self, edgelist):
        for (a,b) in edgelist:
            self.addEdge(a,b)
    
    def getNumNodes(self):
        return len(self.nodes)
    
    def getNumEdges(self):
        return len(self.edges)
    
    def getNodeDegree(self, node):
        d = 0
        for (a,b) in self.edges:
            if a == node:
                d = d + 1
            if b == node:
                d = d + 1
        return d
    
    def getNeighbours(self, node):
        neighbours = []
        for (a,b) in self.edges:
            if a == node:
                neighbours.append(b)
            elif b == node:
                neighbours.append(a)
        return neighbours
    
    def getNodeDegreeList(self):
        '''
        Returns (nl,dl) where nl = list of nodes and dl = corresponding list of degrees of nl
        '''
        nodelist = []
        degreelist = []
        for v in self.nodes:
            nodelist.append(v)
            degreelist.append(self.getNodeDegree(v))
        return (nodelist,degreelist)


    ## In terms of subsetting, it looks like they have a function for getting a subgraph
    ## We might be able to use this to subset the graph in terms of whatever properties
    ## That is, just create a subgraph of neighbors and calculate in the same way
    ## We need to figure out how to create nodelist given the traits
    ## How are the traits stored with the graph? Where can we find node 1's trait?
    ## 
    def getSubgraph(self,nodelist):
        subgph = Graph()
        for x in nodelist:
            subgph.addNode(x)
        for (a,b) in self.edges:
            if a in nodelist and b in nodelist:
                subgph.addEdge(a,b)
        return subgph
    
    def getSplicedGraph(self,partitionList):
        '''
        Given a partition (list of list of nodes), returns the subgraph with
        all edges crossing between partitions discarded
        '''
        spliced = Graph()
        for x in self.nodes:
            spliced.addNode(x)
        for prt in partitionList:
            for (a,b) in self.edges:
                if a in prt and b in prt:
                    spliced.addEdge(a,b)
        return spliced
    
    def print(self):
        print("Graph: {0} vertices, {1} edges".format(len(self.nodes), len(self.edges)) )
        print("Edges: {0} ".format(self.toString()))
    
    def toString(self):
        st ="{ "
        for (a,b) in self.edges:
            st = st +("({0},{1}), ".format(a,b))
        st = st +"}"
        return st
    
    def outputToPajek(self):
        LABEL_PREFIX = "v"
        st = "*Vertices {0}\n".format(len(self.nodes))
        counter = 1
        for v in self.nodes:
            st += " {0} \"{1}{2}\"        0.1  0.1  0.1\n".format(counter,LABEL_PREFIX,v)
            counter += 1
        st += "*Arcs\n"
        for (a,b) in self.edges:
            st += " {0}  {1}  1\n".format(a+1,b+1)
        return st
    
    def toEval(self):
        return self.edges

def graphEval(edgeString, nVertices):
    g = Graph(nVertices)
    g.edges = eval(edgeString)
    return g

def testme():
    # tests
    g = Graph()
    g.addNode(1)
    if g.getNumNodes() != 1 : print("Error #1")
    g.addNode(2)
    g.addNode(3)
    if g.getNumNodes() != 3 : print("Error #2")
    if g.getNumEdges() != 0 : print("Error #3")
    g.addEdge(1,2)
    g.addEdge(2,3)
    if g.getNumEdges() != 2 : print("Error #4")
    g.addNode(4)
    g.addEdge(3,4)
    if g.getNumNodes() != 4 : print("Error #5")
    if g.getNumEdges() != 3 : print("Error #6")
    g.addEdge(1,4)
    gg = g.getSubgraph({1,2,3})
    if gg.getNumNodes() != 3 : print("Error #7")
    if gg.getNumEdges() != 2 : print("Error #8")
    
    #g.addNode(1)
    
    g.print()
    gg.print()

#Graph().testme()
