'''
Created on 2022-08-01

Reading in the twitter graph and returning the structure
To be used as the underlying graph in simulations

The filename of the graph is edgelist.txt and should be saved
in the same folder as the simulation
'''

#print("Started")

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#print("imported")


def produceTwitterGraph():

    G = nx.read_edgelist("edgelist.txt", nodetype = int)

    #print(len(G.edges()))
    #print(len(G.nodes()))


    return G


graph = produceTwitterGraph()
