import threading
import time
#create a class

class myThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID=threadID
    def run(self):
        print("This is thread - "+str(self.threadID))
        time.sleep(1)
        print("Thread"+str(self.threadID)+"terminates!")

# Create five threads
threads=[]
for i in range(5):
    threadTemp=myThread(i)
    threadTemp.start()
    threads.append(threadTemp)

# Wait for all threads to be completed.
for i in threads:
    i.join()

print("Exit main thread!")