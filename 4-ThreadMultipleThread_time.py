#This example illustrate how to use time.sleep() function
import threading
import time

#create a class

class myThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID=threadID
    def run(self):
        print("This is thread - "+str(self.threadID))
        #The time.sleep(t) function takes one argument "t", indicates the number of seconds 
        #an execution to be suspended
        time.sleep(self.threadID)  
        print("Thread"+str(self.threadID)+" terminates!")
        
#create 5 threads, the number of seconds that each thread to be suspended are 1, 2, 3, 4, 5
for i in range(1,6):
    threadTemp=myThread(i)
    threadTemp.start()
    
print("Exit main thread!")