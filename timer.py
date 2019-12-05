import time

def timer(f):
    def wrapper(*args, **kargs):
        t0 = time.time()
        f(*args, **kargs)
        t1 = time.time()
        lapse = t1 - t0
        print("{seconds:.1f} seconds used.".format(seconds=lapse))
    return wrapper