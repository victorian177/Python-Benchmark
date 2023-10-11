# This example illustrate erroneous of inappropriate coding of two threads with a shared variable.

import threading

class myThread():
    #Global variable count
    a=[]
    for i in range(20):
        a.append(i)
    #uncomment next line to make the code correct
    #threadLock=threading.Lock()
    def getItem(self, tName):
        while 1:
            #uncomment next line to make the code correct
            #self.threadLock.acquire()
            if (len(self.a)==0):
                break
            value=self.a[0]
            self.a.remove(value)
            print("Thread "+tName+", remove "+str(value))
            #uncomment next line to make the code correct
            #self.threadLock.release()
            
    # Two threads are constantly removing elements from a shared array of numbers.    
    def run(self):
        thread1=threading.Thread(target=self.getItem,args=["1"])
        thread2=threading.Thread(target=self.getItem,args=["2"])
        thread1.start()
        thread2.start()

thread1=myThread()
thread1.run()
