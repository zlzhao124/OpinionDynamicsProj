'''
Created on Feb 25, 2013

@author: akhtsang

Creates an objects that contains a number of items that can be sampled from (without replacement), 
and a weight for each item.  Once an item is sampled, all of its copies are removed from the list.
'''

import random

class WeightedSampler(object):

    def __init__(self, itemlist, weightlist):
        if len(itemlist) != len(weightlist):
            raise Exception("WeightedSampler must operate on lists of the same size")
        self.rafflebox = []
        for (item,weight) in zip(itemlist,weightlist):
            for i in range(weight):
                self.rafflebox.append(item)
        
    def isEmpty(self):
        return len(self.rafflebox) == 0
    
    def _removeFromRaffle(self,r):
        self.rafflebox = [x for x in self.rafflebox if x != r]
    
    def sample(self):
        result = random.sample(self.rafflebox,1)[0]
        self._removeFromRaffle(result)
        return result

    def _testme(self):
        s = WeightedSampler(['a','b','c'],[9,4,1])
        print(s.sample())
        print(s.sample())
        print(s.sample())
        print(s.isEmpty())

#WeightedSampler([],[])._testme()