import signal  
import time  
 
# Our signal handler
def signal_handler(signum, frame):  
    print("Signal Number:", signum, " Frame: ", frame)  
 
def exit_handler(signum, frame):
    print('Exiting....')
    exit(0)
 
# Register our signal handler with `SIGINT`(CTRL + C)
signal.signal(signal.SIGINT, signal_handler)
 
# Register the exit handler with `SIGTSTP` (Ctrl + Z)
signal.signal(signal.SIGTSTP, exit_handler)
 
# While Loop
while 1:  
    print("Press Ctrl + C") 
    time.sleep(3) 
