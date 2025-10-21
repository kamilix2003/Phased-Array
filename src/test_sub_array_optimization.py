import numpy as np
import pytest
from sub_array_optimization import calculate_element_positions

def test_calculate_element_positions_basic():
  # 2 sub-arrays, 2 elements each, major=0.5, minor=0.3
  result = calculate_element_positions(num_of_sub_arrays=2, sub_array_size=2, major_spacing=0.5, minor_spacing=0.3)
  expected = np.array([0.0, 0.3, 0.5, 0.8])
  np.testing.assert_allclose(result, expected, rtol=1e-8, atol=0)


def test_single_element_sub_arrays():
  # 3 sub-arrays, 1 element each, major spacing moves by 1.0
  result = calculate_element_positions(num_of_sub_arrays=3, sub_array_size=1, major_spacing=1.0, minor_spacing=0.2)
  expected = np.array([0.0, 1.0, 2.0])
  np.testing.assert_allclose(result, expected)


def test_zero_sub_array_size_returns_empty():
  # zero elements per sub-array -> empty array
  result = calculate_element_positions(num_of_sub_arrays=4, sub_array_size=0, major_spacing=0.5, minor_spacing=0.1)
  assert isinstance(result, np.ndarray)
  assert result.size == 0


def test_zero_num_of_sub_arrays_returns_empty():
  # zero sub-arrays -> empty array
  result = calculate_element_positions(num_of_sub_arrays=0, sub_array_size=3, major_spacing=0.5, minor_spacing=0.1)
  assert isinstance(result, np.ndarray)
  assert result.size == 0
  def test_calculate_element_positions_basic():
    # 2 sub-arrays, 2 elements each, major=0.5, minor=0.3
    result = calculate_element_positions(num_of_sub_arrays=2, sub_array_size=2, major_spacing=0.5, minor_spacing=0.3)
    expected = np.array([0.0, 0.3, 0.5, 0.8])
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=0)


  def test_single_element_sub_arrays():
    # 3 sub-arrays, 1 element each, major spacing moves by 1.0
    result = calculate_element_positions(num_of_sub_arrays=3, sub_array_size=1, major_spacing=1.0, minor_spacing=0.2)
    expected = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(result, expected)


  def test_zero_sub_array_size_returns_empty():
    # zero elements per sub-array -> empty array
    result = calculate_element_positions(num_of_sub_arrays=4, sub_array_size=0, major_spacing=0.5, minor_spacing=0.1)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


  def test_zero_num_of_sub_arrays_returns_empty():
    # zero sub-arrays -> empty array
    result = calculate_element_positions(num_of_sub_arrays=0, sub_array_size=3, major_spacing=0.5, minor_spacing=0.1)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


  def test_get_all_phase_shifts_small():
    # num_elements=2, num_of_bits=1 -> 2^1 per element, total rows = 4
    result = get_all_phase_shifts(num_elements=2, num_of_bits=1)
    expected = np.array([[0, 0],
                         [1, 0],
                         [0, 1],
                         [1, 1]], dtype=int)
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result, expected)


  def test_get_all_phase_shifts_three_elements_two_bits():
    # num_elements=3, num_of_bits=2 -> (2**2)**3 = 64 rows, values in [0,3]
    result = get_all_phase_shifts(num_elements=3, num_of_bits=2)
    assert result.shape == ((2**2)**3, 3)
    assert np.issubdtype(result.dtype, np.integer)
    assert np.all((result >= 0) & (result <= 3))
    # all rows should be unique, first row all zeros, last row all max (3)
    assert np.unique(result, axis=0).shape[0] == result.shape[0]
    assert np.all(result[0] == 0)
    assert np.all(result[-1] == 3)


  def test_get_all_phase_shifts_zero_elements_returns_empty():
    # zero elements -> empty 2D array
    result = get_all_phase_shifts(num_elements=0, num_of_bits=4)
    assert isinstance(result, np.ndarray)
    assert result.size == 0
    assert result.shape == (0, 0)
