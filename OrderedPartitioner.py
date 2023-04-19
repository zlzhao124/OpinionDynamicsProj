'''
Created on 2013-02-11

@author: Alan

This class is an iterator that iterates over all possible partitions of 
a given list of elements (interpreted as a LIST).  Partitions are duplicated 
with different orderings.  This class was mainly a prototype 
for the UnorderePartitioner class 
'''
from NBoolIterator import NBoolIterator

class OrderedPartitioner(object):
    

    def __init__(self, componentList):
        self.componentList = componentList
        self.componentLen = len(componentList)
        self.done = False
        self._setup()
    
    def __iter__(self):
        return self
    
    def _setup(self):
        if self.componentLen == 0:
            self.done = True
        else:
            self.nbiter = NBoolIterator(self.componentLen)
            # need to next it once to avoid an empty partition
            next(self.nbiter)
            (self.firstComponent,rest) = self._split(next(self.nbiter),self.componentList)
            self.riter = OrderedPartitioner(rest)
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
            self.riter = OrderedPartitioner(remnants)
            self.restComponents = next(self.riter)
        
        return
    
    def __next__(self):
        if self.done: raise StopIteration
        self._advance()
        return [self.firstComponent] + self.restComponents
        
            
    def print(self):
        print(self.firstComponent)
    
    
# Testing!
'''
prt = OrderedPartitioner([])
print(prt._split([True,True,False],[1,1,2]))
print(prt._split([True,True,True],"AAA"))
print(prt._split([False,False,False],"BBB"))
print(prt._split([False,True,False],"BAB"))
'''

prt = OrderedPartitioner([1,2,3])
#prt.print()
for x in prt:
    print(x)
