import sys
import os
from box_maker import make_blurred_box

# testing a 3 by 3 by 3 box
def test_around_center():
  box_size = 3
  prebox = [[1, 1, 1, 15]]
  center_prob = 0.44

  box = make_blurred_box(prebox, center_prob, box_size)

  # center
  assert box[1, 1, 1, 15] == 0.44

  # major axis
  assert box[1, 1, 0, 15] == 0.04
  assert box[0, 1, 1, 15] == 0.04
  assert box[1, 0, 1, 15] == 0.04
  assert box[2, 1, 1, 15] == 0.04
  assert box[1, 2, 1, 15] == 0.04
  assert box[1, 1, 2, 15] == 0.04

  # edge
  assert box[0, 1, 0, 15] == 0.02
  assert box[0, 2, 1, 15] == 0.02
  assert box[1, 0, 2, 15] == 0.02
  assert box[2, 1, 0, 15] == 0.02

  # corner
  assert box[0, 0, 0, 15] == 0.01
  assert box[2, 2, 2, 15] == 0.01
  assert box[2, 0, 0, 15] == 0.01
  assert box[2, 0, 2, 15] == 0.01

def test_around_major_axes():
  box_size = 3
  prebox = [[1, 1, 0, 15]]
  center_prob = 0.44

  box = make_blurred_box(prebox, center_prob, box_size)

  # center
  assert box[1, 1, 0, 15] == 0.44

  # major axis
  assert box[1, 2, 0, 15] == 0.04
  assert box[0, 1, 0, 15] == 0.04
  assert box[2, 1, 0, 15] == 0.04
  assert box[1, 1, 1, 15] == 0.04

  # edge
  assert box[0, 2, 0, 15] == 0.02
  assert box[0, 1, 1, 15] == 0.02
  assert box[1, 0, 1, 15] == 0.02
  assert box[2, 1, 1, 15] == 0.02

  # corner
  assert box[0, 2, 1, 15] == 0.01
  assert box[2, 2, 1, 15] == 0.01
  assert box[0, 0, 1, 15] == 0.01
  assert box[2, 0, 1, 15] == 0.01

def test_around_edges():
  box_size = 3
  prebox = [[0, 1, 0, 15]]
  center_prob = 0.44

  box = make_blurred_box(prebox, center_prob, box_size)

  # center
  assert box[0, 1, 0, 15] == 0.44

  # major axis
  assert box[0, 2, 0, 15] == 0.04
  assert box[0, 0, 0, 15] == 0.04
  assert box[0, 1, 1, 15] == 0.04
  assert box[1, 1, 0, 15] == 0.04

  # edge
  assert box[1, 2, 0, 15] == 0.02
  assert box[0, 0, 1, 15] == 0.02
  assert box[0, 2, 1, 15] == 0.02
  assert box[1, 1, 1, 15] == 0.02

  # corner
  assert box[1, 2, 1, 15] == 0.01
  assert box[1, 0, 1, 15] == 0.01


def test_around_corners():
  box_size = 3
  prebox = [[0, 0, 0, 15]]
  center_prob = 0.44

  box = make_blurred_box(prebox, center_prob, box_size)

  # center
  assert box[0, 0, 0, 15] == 0.44

  # major axis
  assert box[1, 0, 0, 15] == 0.04
  assert box[0, 1, 0, 15] == 0.04
  assert box[0, 0, 1, 15] == 0.04

  # edge
  assert box[1, 1, 0, 15] == 0.02
  assert box[0, 1, 1, 15] == 0.02
  assert box[1, 0, 1, 15] == 0.02

  # corner
  assert box[1, 1, 1, 15] == 0.01

def test_diff_center_probs():
  box_size = 3
  prebox = [[1, 1, 1, 15]]

  # check values in major axes
  box = make_blurred_box(prebox, 0.8, box_size)
  assert abs(box[1, 1, 0, 15] - (0.2/14)) < 0.0001

  box = make_blurred_box(prebox, 0.2, box_size)
  assert abs(box[1, 1, 0, 15] - (0.8/14)) < 0.0001

  box = make_blurred_box(prebox, 0, box_size)
  assert abs(box[1, 1, 0, 15] - (1/14)) < 0.0001

  box = make_blurred_box(prebox, 1, box_size)
  assert box[1, 1, 0, 15] == 0


