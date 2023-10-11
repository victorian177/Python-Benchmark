'''
This script domonstrates how to use a timer in a sperate thread to control the loop
'''

import threading
import time

class myThread():
    __terminate=False

    def __init__(self, timer):
        self.__timer=timer

    def count(self):
        i=0
        while not self.__terminate:
            i+=1
            time.sleep(1)
        print("Thread terminated, count to "+str(i))


    def timer(self):
        time.sleep(self.__timer)
        self.__terminate=True

    def run(self):
        t1=threading.Thread(target=self.count)
        t2=threading.Thread(target=self.timer)
        t1.start()
        t2.start()

thread1=myThread(30)
thread1.run()

thread2=myThread(5)
thread2.run()
