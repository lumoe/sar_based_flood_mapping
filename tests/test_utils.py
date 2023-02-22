import unittest

from utils import paths_train_reference_images

def test_paths_train_reference_images_returns_dictionary_water():
    result = paths_train_reference_images(type="water")
    assert isinstance(result, tuple)

def test_paths_train_reference_images_returns_dictionary_flood():
    result = paths_train_reference_images(type="flood")
    assert isinstance(result, tuple)