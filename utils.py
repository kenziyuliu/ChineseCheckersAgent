import datetime
from collections import Mapping, Container
from sys import getsizeof


def cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stress_message(message, extra_newline=False):
    print('{2}{0}\n{1}\n{0}{2}'.format('='*len(message), message, '\n' if extra_newline else ''))


def deepsizeof(obj, visited):
    d = deepsizeof
    if id(obj) in visited:
        return 0

    r = getsizeof(obj)
    visited.add(id(obj))

    if isinstance(obj, Mapping):
        r += sum(d(k, visited) + d(v, visited) for k, v in obj.items())

    if isinstance(obj, Container):
        r += sum(d(x, visited) for x in obj)

    return r


