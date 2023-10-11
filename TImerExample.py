# This example illustrate how to setup a timer to stop the execution of a thread
import threading
import time

class myThread(threading.Thread):
    def __init__(self, ThreadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID=ThreadID
        self.name=name
        self.counter=counter
        #A flag for terminating the thread
        self.flag=False
        
    def exit(self):
        self.flag=True
    
    def run(self):
        print("Starting "+ self.name)
        self.print_time(self.name, 5, self.counter)
        print("Exiting "+self.name)
    
    def print_time(self, threadName, counter, delay):
        while not self.flag:
        #counter:
           # if self.flag:
           #     break
            time.sleep(delay)
            print("%s: %s" % (threadName, time.ctime(time.time())))
            #counter -= 1
#Timer
def countDown(timer):
    time.sleep(timer)
    thread1.exit()
    thread2.exit()
    
thread1=myThread(1, "Thread-1", 1)
thread2=myThread(2,"Thread-2", 2)
thread3=threading.Thread(target=countDown,args=[5])

thread1.start()
thread2.start()
thread3.start()
