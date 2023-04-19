'''
Created on 2013-02-12

@author: Alan (revised by Zach)
'''
from Graph import Graph
from DistanceMatrix import DistanceMatrix
import random
import math
from WeightedSampler import WeightedSampler
import networkx as nx
import numpy as np

def makeClique(size):
    g = Graph(size)
    for i in range(size):
        for j in range(i+1,size):
            g.addEdge(i,j)
    return g

def makeLine(size):
    g = Graph(size)
    for i in range(size-1):
        g.addEdge(i,i+1)
    return g

def makeBipartite(size_a, size_b):
    g = Graph(size_a+size_b)
    for i in range(size_a):
        for j in range(size_a,size_a+size_b):
            g.addEdge(i,j)
    return g

def makeRandomGraph(size,p,ConnectedOnly=False):
    '''
    Generates random graph according to the Erdos-Renyi model (all possible edges are added with uniform
    probability p).
    '''
    g = Graph(size)
    for i in range(size):
        for j in range(i+1,size):
            if random.random() <= p:
                g.addEdge(i,j)
    if ConnectedOnly == False or DistanceMatrix(g).isConnected():
        return g
    else:
        return makeRandomGraph(size, p, ConnectedOnly)

def makeHomophilyERGraph(size,p,opinions,ConnectedOnly=True):
    '''
    Generates random graph based on the Erdos-Renyi model, with connection probabilities weighted by
    similarity
    '''
    g = Graph(size)
    for i in range(size):
        for j in range(i+1,size):
            od = abs(opinions.getWeight(i)-opinions.getWeight(j))
            if random.random() <= p*(1-od):
                g.addEdge(i,j)
    if ConnectedOnly == False or DistanceMatrix(g).isConnected():
        print("Graph is disconnected.  Regenerating.")
        return g
    else:
        return makeRandomGraph(size, p, ConnectedOnly)

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

def makeHomophilyBAGraph(size,initial_graph,m,opinions,bandwidth,conviction,ConnectedOnly=True):
    '''
    Generates a random graph using a variation of the Barabasi-Albert model, where the attachment 
    chance is weighted by the degree of similarity between vertex opinions 
    '''
    PRECISION = 1000
    g = initial_graph
    while g.getNumNodes() < size:
        newlabel = g.getNumNodes()
        g.addNode(newlabel)
        (nl,wl) = g.getNodeDegreeList() # gets node list and degree list
        dol = opinions.getDiffList(opinions.getWeight(newlabel))  # gets list of opinion differences
        # normalize dol weights by trust kernel (without conviction)
        #dol = [max(0, math.exp(- (d**2) / bandwidth)) for d in dol]
        # normalize dol weights by trust kernel
        dol = [max(0, math.exp(- (d**2) / bandwidth) - conviction) for d in dol]
        wl = [round(PRECISION*a*b) for a,b in zip(wl,dol)]
        sampler = WeightedSampler(nl,wl)
        if sampler.isEmpty():
            rand_node = random.randint(0, newlabel-1)
            g.addEdge(newlabel, rand_node)
        for i in range(m):
            if sampler.isEmpty():
                break
            g.addEdge(newlabel,sampler.sample())
    if ConnectedOnly == True and not DistanceMatrix(g).isConnected():
        print("Graph is disconnected.  Regenerating.")
        return makeHomophilyBAGraph(size,initial_graph,m,opinions,bandwidth,conviction)
    return g

def twitterGraphAttempt(size):
    '''
    Generates a twitter graph using the same structure as the Twitter graph from the Musco data. 
    This data has 548 nodes, so n=548 for all of these simulations.
    '''
    g = nx.read_edgelist("twitter_edgelist.csv", delimiter=",", nodetype=int)
    #print(g.nodes())
    #node_list = set()
    #edge_list = set()

    g_attempt = Graph(size)

    #for node in g.nodes():
    #node_list.add(node)
    #g_attempt.addNode(node)

    for edge in g.edges():
    #edge_list.add(edge)
    #print(edge[0], edge[1])
        g_attempt.addEdge(edge[0], edge[1])

    #print(g_attempt.edges)
    #print(g_attempt.nodes)

    return g_attempt

def facebookGraphAttempt(size):
    '''
    Generates a graph using ego network with ego 0 information
    This data has 347 nodes, so n=347 for all of these simulations.
    '''
    g = nx.read_edgelist("EgoNodeData/0edges.csv", delimiter=",", nodetype=int)
    g_attempt = Graph(size)

    for edge in g.edges():
        g_attempt.addEdge(edge[0], edge[1])

    return g_attempt


def senateGraphAttempt(size):
    '''
    Generates a graph using senator information
    This data has 100 nodes, so n=100 for all of these simulations.
    '''
    g = nx.read_edgelist("senate_edges.csv", delimiter=",", nodetype=int)
    g_attempt = Graph(size)

    for edge in g.edges():
        g_attempt.addEdge(edge[0], edge[1])

    return g_attempt



def egoGraph(size):
    '''
    Generates an ego graph using the same structure as the Facebook graph from the Stanford(?) data. 
    This data has 348 nodes, so n=4038 for all of these simulations.
    '''
    g_read = nx.read_edgelist("ego_edgelist.csv", delimiter=",", nodetype=int)

    g = Graph(size)

    for edge in g_read.edges():
        g.addEdge(edge[0], edge[1])

    return g


def testme():
    gi = Graph(2)
    gi.addEdge(1,0)
    g = makeRandomBAGraph(8,gi,1)
    g.print()
    
#testme()