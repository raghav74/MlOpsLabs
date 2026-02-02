import pytest
from src import data_processor

def test_normalize():
    assert data_processor.normalize([0, 50, 100]) == [0.0, 0.5, 1.0]
    assert data_processor.normalize([10, 10, 10]) == [0.0, 0.0, 0.0]
    assert data_processor.normalize([1, 2, 3, 4, 5]) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert data_processor.normalize([-10, 0, 10]) == [0.0, 0.5, 1.0]


def test_standardize():
    result = data_processor.standardize([1, 2, 3, 4, 5])
    assert abs(sum(result)) < 1e-10  # mean should be ~0
    assert data_processor.standardize([5, 5, 5]) == [0.0, 0.0, 0.0]
    

def test_fill_missing():
    assert data_processor.fill_missing([1, None, 3]) == [1, 0, 3]
    assert data_processor.fill_missing([None, None], fill_value=-1) == [-1, -1]
    assert data_processor.fill_missing([1, 2, 3]) == [1, 2, 3]
    assert data_processor.fill_missing([]) == []


def test_compute_statistics():
    stats = data_processor.compute_statistics([1, 2, 3, 4, 5])
    assert stats["mean"] == 3.0
    assert stats["min"] == 1
    assert stats["max"] == 5
    assert stats["count"] == 5
    
    stats2 = data_processor.compute_statistics([10, 20, 30])
    assert stats2["mean"] == 20.0
    assert stats2["min"] == 10
    assert stats2["max"] == 30
