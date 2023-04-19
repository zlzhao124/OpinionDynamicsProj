'''
Created on 2013-02-11

@author: Alan

This creates an iterator that iterates over all possible combinations of 
N boolean values, and returns each successive one as a N-tuple of bools
'''
class NBoolIterator(object):
    

    def __init__(self, n):
        self.n = n
        self.iterCounter = 0
        self.n = n
        self.iterLimit = 2**n
    
    def __iter__(self):
        return self
     
    '''       
    def __next__(self):
        count = self.iterCounter
        nbool = [];
        for i in range(self.n):
            nbool.append(count%2==1)
            count = int(count / 2)
        #print(nbool)
        yield nbool
        self.iterCounter = self.iterCounter + 1
        '''
    def __next__(self):
        return self.next()
    
    def next(self):
        if self.iterCounter == self.iterLimit:
            raise StopIteration
        
        #return reversed(_toBools(("{0:"+str(self.n)+"b}").format(self.iterCounter-1)))
        templist = _toBools(("{0:b}").format(self.iterCounter))
        templist = [False] * (self.n - len(templist)) + templist
        self.iterCounter = self.iterCounter + 1
        return templist
        
#        count = self.iterCounter
#        nbool = [];
#        for i in range(self.n):
#            nbool.append(count%2==1)
#            count = int(count / 2)
#        self.iterCounter = self.iterCounter + 1
#        return nbool

def _toBools(l):
    return [_isTrue(i) for i in l]

def _isTrue(b):
    return b == '1'
        
def testme():
    nbi = NBoolIterator(5)
    for x in nbi:
        print([i for i in x])
    try:
        next(nbi)
    except StopIteration:
        print("End of iterations")

# testing
#testme()