# -*- coding: utf8 -*-

#
import inspect


def isdebugging():
    """See Also https://stackoverflow.com/questions/333995/how-to-detect-that-python-code-is-being-executed-through-the-debugger"""
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False
