import sys
import os
from train_test import normalize_classes

def test_in_order():
  centers = [0, 1, 1, 1, 1, 1, 2, 2, 2]
  expected_weights = {0:3.0, 1:(3/5), 2:1.0}

  assert normalize_classes(centers) == expected_weights

def test_out_of_order():
  centers = [1, 1, 1, 1, 1, 0, 2, 2, 2]
  expected_weights = {0:3.0, 1:(3/5), 2:1.0}

  assert normalize_classes(centers) == expected_weights

def test_one_value():
  centers = [1]
  expected_weights = {1:1.0}

  assert normalize_classes(centers) == expected_weights

def test_one_class():
  centers = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  expected_weights = {1:1.0}

  assert normalize_classes(centers) == expected_weights

def test_no_values():
  centers = []
  expected_weights = {}

  assert normalize_classes(centers) == expected_weights








