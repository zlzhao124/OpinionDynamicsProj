'''
Created on 2013-02-12

@author: Alan
'''
from NBoolIterator import NBoolIterator

class Partitions(object):
    '''
    classdocs
    '''

    def __init__(self, itemslist,skipEmpty=False):
        self.items = itemslist
        self.nbiter = NBoolIterator(len(itemslist))
        if skipEmpty:
            next(self.nbiter)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        nbools = next(self.nbiter)
        result = []
        for (b,v) in zip(nbools,self.items):
            if b: result.append(v)
        return tuple(result)
    

def _testme():
    p = Partitions([1,2,3])
    for x in p:
        print(x)

#_testme()