import time

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

@measure_execution_time
def sample_function():
    time.sleep(2)
    return "Function completed"

sample_function()
