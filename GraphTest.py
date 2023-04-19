'''
Created on 2022-08-15

@author: Jean

Compare the graph generated in the code to the graph from the Twitter data to make sure
'''

#Import scripts we've written
import GraphGen
from Graph import Graph
from createTwitterGraph import produceTwitterGraph

#Import the other stuff
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Everything Imported")


g_generated = GraphGen.makeRandomGraph(size = 200, p = 4, ConnectedOnly=True)

print("Graph 1 Generated")

g_twitter = GraphGen.twitterGraphAttempt(size = 548)

print("Graph 2 Generated")

## Compare the graphs

print(g_generated.edges)
print(g_twitter.edges)


print("Script Complete")
