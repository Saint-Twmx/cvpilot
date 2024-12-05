def log_time(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 花费 {end_time - start_time} seconds")
        return value
    return wrapper