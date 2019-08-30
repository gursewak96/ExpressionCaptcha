import time

t_start = time.time()
t_end = t_start

def restart():
    global t_start, t_end
    t_start = time.time()
    t_end = t_start
    return int(t_end - t_start)

def getCount():
    global t_start, t_end
    t_end = time.time()
    return int(t_end - t_start)
