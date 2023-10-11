'''
This script demonstrates how to use time.sleep function
'''

import threading
import time

class myThread(threading.Thread):
    __terminate=False
    __count=0

    def terminate(self):
        self.__terminate=True
        
    def count(self):
        while not self.__terminate:
            self.__count+=1
            time.sleep(1)
        print("Thread terminated, count to "+str(self.__count))

    def display(self):
        print(self.__count)
        
    def run(self):
        self.count()


t1=myThread()
t1.start()
for i in range(5):
    t1.display()
    time.sleep(2)
t1.terminate()
