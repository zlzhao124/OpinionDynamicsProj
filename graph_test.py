from Graph import Graph
import GraphGen
from DistanceMatrix import DistanceMatrix
import random
import math
from WeightedSampler import WeightedSampler
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

g = nx.read_edgelist("edgelist_reformatted.csv", delimiter=",", nodetype=int)
#print(g.nodes())
#node_list = set()
#edge_list = set()

g_attempt = Graph(548)

#for node in g.nodes():
    #node_list.add(node)
    #g_attempt.addNode(node)

for edge in g.edges():
    #edge_list.add(edge)
    #print(edge[0], edge[1])
    g_attempt.addEdge(edge[0], edge[1])

#print(edge_list)
#print(node_list)

print(g_attempt.edges)
print(g_attempt.nodes)

def makeRandomBAGraph(size,initial_graph,m):
    '''
    Generates scale-free random graph according to the Barabasi-Albert model of given size.
    Begins with initial_graph and adds nodes one at a time, of degree m, connecting to existing 
    nodes with probability proportional to their current degrees
    '''
    g = initial_graph
    while g.getNumNodes() < size:
        newlabel = g.getNumNodes()
        g.addNode(newlabel)
        (nl,wl) = g.getNodeDegreeList() 
        sampler = WeightedSampler(nl,wl)
        for i in range(m):
            if sampler.isEmpty():
                break
            g.addEdge(newlabel,sampler.sample())

    return g

g1 = makeRandomBAGraph(40, GraphGen.makeClique(2), 4)
#print(g1.nodes)
#print(g1.edges)
