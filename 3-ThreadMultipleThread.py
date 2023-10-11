import threading

#create a class

class myThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID=threadID
    def run(self):
        print("This is thread - "+str(self.threadID))
        

for i in range(5):
    threadTemp=myThread(i)
    threadTemp.start()
