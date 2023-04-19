'''
Created on 2013-02-11

@author: Alan

This class is an iterator that iterates over all possible partitions of 
a given list of elements (interpreted as a set).  No partitions are repeated.
Assumes the list has no duplicates
'''
from NBoolIterator import NBoolIterator

class Partitioner(object):
    
    def __init__(self, componentList):
        self.componentList = componentList.copy()
        self.componentLen = len(componentList)
        self.done = False
        self._setup()
    
    def __iter__(self):
        return self
    
    def _setup(self):
        if self.componentLen == 0:
            self.done = True
        else:
            self.specialComponent = self.componentList.pop()
            #self.componentList = self.componentList[1::]
            self.nbiter = NBoolIterator(self.componentLen)
            (self.firstComponent,rest) = self._split(next(self.nbiter),self.componentList)
            self.riter = Partitioner(rest)
            self.restComponents = []
            
    
    def _split(self,inclusionList, componentList):
        l1 = []
        l2 = []
        for (b,v) in zip(inclusionList,componentList):
            if b: l1.append(v)
            else: l2.append(v)
        return (l1,l2)
    
    def _advance(self):
        if self.componentLen == 1:
            self.done = True
            return
        
        try:
            self.restComponents = next(self.riter)
        except StopIteration:
            (self.firstComponent,remnants) = self._split(next(self.nbiter),self.componentList)
            if remnants == [] :
                self.done = True
                self.restComponents = []
                return
            self.riter = Partitioner(remnants)
            self.restComponents = next(self.riter)
        
        return
    
    def __next__(self):
        if self.done: raise StopIteration
        self._advance()
        return [sorted([self.specialComponent]+self.firstComponent)] + self.restComponents
        #return [[self.specialComponent]+self.firstComponent] + self.restComponents
        
            
    def print(self):
        print(self.firstComponent)
    
def testme():
    prt = Partitioner([1,2,3,4])
    #prt.print()
    for x in prt:
        print(x)

# Testing!

