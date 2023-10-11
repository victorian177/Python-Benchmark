# Three methods to create a thread.
import threading

#Thread example: function without args

def threadFunction1():
    print("This is thread 1\n ")

#Create a thread
thread1=threading.Thread(target=threadFunction1)

#Start the thread
thread1.start()


#Thread example: function with args

def threadFunction2(name):
    print("This is thread "+name)
    
#create a thread
thread2=threading.Thread(target=threadFunction2, args=["thread2"])

thread2.start()

#Create thread in class
class myThread(threading.Thread):
    #override run function
    def run(self):
        print("Thread - 3")
        
#Create a new thread:
thread3=myThread()
thread3.start()
