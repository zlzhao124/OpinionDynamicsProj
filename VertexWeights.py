'''
Created on 2013-03-31

@author: Alan

Updated on 2022-11-07 to include types dynamics. Added setType and getType. getType is very straightforward, and returns the 
type of the node input. setType takes the numeric opinion and returns a type based on that value (Bernouilli trial)
'''
import random

class VertexWeights(object):

    def __init__(self, nVertices, initial_value=0):
        '''
        Assumes vertex labels are integers starting at 0
        '''
        self.weights = [initial_value]*nVertices
        self.types = [initial_value]*nVertices
        
    def getIterator(self):
        return range(self.getNumNodes())
    
    def getNumNodes(self):
        return len(self.weights)
        
    def getWeight(self, nodeID):
        return self.weights[nodeID]
    
    def setWeight(self, nodeID, val):
        self.weights[nodeID] = val

    def getType(self, nodeID):
        return self.types[nodeID]
    
    def setType(self, nodeID, val):
        trial = random.random()
        type_assign = 0
        if trial < val: 
          type_assign = 1
        self.types[nodeID] = type_assign
        
    def getDiffList(self, centralVal):
        '''
        Returns a list of weights, each entry i is the difference of self.weights[i] from centralVale
        '''
        dl = []
        for x in self.weights:
            dl.append(abs(x - centralVal))
        return dl
        
    def outputToPajek(self):
        st = "*Vertices {0}\n".format(len(self.weights))
        for w in self.weights:
            st += "{0}\n".format(w)
        return st

def main():
    vw = VertexWeights(5)
    op = VertexWeights(5)
    vw.setWeight(0, 0)
    vw.setWeight(1, 0.24)
    vw.setWeight(2, -0.3)
    vw.setWeight(3, 0.6)
    vw.setWeight(4, 1)


    op.setType(0, 0)
    op.setType(1, 0.24)
    op.setType(2, -0.3)
    op.setType(3, 0.6)
    op.setType(4, 1)
    dl = vw.getDiffList(0.5)
    print(vw.weights)
    print(op.types)
    print(vw.types[4])
    print(vw.getType(4))
    print(op.getType(4))

main()