# A fun program to sort a list of numbers using multithreads
import threading
import time

class myThread():
    __results=[]
    __threadList=[]
    
    def __init__(self,inputList):
        self.input=inputList
        
    threadLock=threading.Lock()
    def foo(self, inputNum):
        time.sleep(inputNum)
        self.threadLock.acquire()
        self.__results.append(inputNum) 
        self.threadLock.release()
        
    def display(self):
        for item in self.__results:
            print(item)
            
    def run(self):
        for n in self.input:
            tempTread=threading.Thread(target=self.foo, args=[n])
            tempTread.start()
            self.__threadList.append(tempTread)
            
    def join(self):
        for t in self.__threadList:
            t.join()

thread1=myThread([3,2,1,6,3,4,8])
thread1.run()
thread1.join()
thread1.display()