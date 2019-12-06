import os, sys, time

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(current_dir)
sys.path.insert(1, temp)

import numpy as np
import pytest
from LRST.low_rank_statistics import information_criterion, AIC_criterion, BIC_criterion

def test_information_criterion():
    s = np.array([2, 0.5, 0.1])
    ic = information_criterion(s)
    assert ic == pytest.approx((0.294, 0), rel=1e-3)

def test_AIC_rank_estimation():
    s = np.array([2, 0.5, 0.1])
    criterion = AIC_criterion(s, n=25)
    assert criterion == pytest.approx((19.7, 8), rel=1e-1)

def test_BIC_rank_estimation():
    s = np.array([2, 0.5, 0.1])
    criterion = BIC_criterion(s, n=25)
    assert criterion == pytest.approx((45.49, 25.75), rel=1e-1)