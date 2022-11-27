import functools


def limited_calls(semaphore_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with getattr(self, semaphore_name):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
