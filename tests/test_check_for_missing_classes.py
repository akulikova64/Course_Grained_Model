import sys
import os
import collections
from train_test import check_for_missing_classes
from train_test import AA_DICT

def test_one_missing():
  centers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  centers_dict = collections.Counter(centers)
  missing_aa = check_for_missing_classes(centers_dict, AA_DICT)
 
  assert missing_aa == ["C"]

def test_three_missing():
  centers = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  centers_dict = collections.Counter(centers)
  missing_aa = check_for_missing_classes(centers_dict, AA_DICT)
  print(missing_aa)
 
  assert missing_aa == ["R", "A", "C"]

def test_all_missing():
  centers = []
  centers_dict = collections.Counter(centers)
  missing_aa = check_for_missing_classes(centers_dict, AA_DICT)
  print(missing_aa)
 
  assert missing_aa == ['H', 'E', 'D', 'R', 'K', 'S', 'T', 'N', 'Q', 'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W', 'P', 'G', 'C']

def test_none_missing():
  centers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  centers_dict = collections.Counter(centers)
  missing_aa = check_for_missing_classes(centers_dict, AA_DICT)
  print(missing_aa)
 
  assert missing_aa == []

def test_out_of_order():
  centers = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  centers_dict = collections.Counter(centers)
  missing_aa = check_for_missing_classes(centers_dict, AA_DICT)
  print(missing_aa)
 
  assert missing_aa == ['C']