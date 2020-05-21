import sys
import os
sys.path.append(os.path.abspath(".."))

import rotations
# testing the rotation functions in rotations.py 

def test():
  assert (1, 2, 3) == (1, 2, 3)

def test_2():
  assert (1, 2, 3) == (3, 2, 1)