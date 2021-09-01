import os
import sys

_current_dir = os.path.abspath(os.path.dirname(__file__))
if sys.path and os.path.abspath(sys.path[0]) != _current_dir:
    sys.path.insert(0, _current_dir)
del _current_dir
