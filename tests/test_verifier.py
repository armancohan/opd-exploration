"""Tests for math answer verifier."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.verifier import extract_boxed_answer, normalize_math_answer, verify_math_answer, batch_verify


def test_extract_simple_number():
    assert extract_boxed_answer(r"The answer is \boxed{42}") == "42"


def test_extract_fraction():
    result = extract_boxed_answer(r"So we get \boxed{\frac{3}{4}}")
    assert result == r"\frac{3}{4}"


def test_extract_nested_braces():
    result = extract_boxed_answer(r"\boxed{\frac{a+b}{c+d}}")
    assert result == r"\frac{a+b}{c+d}"


def test_extract_last_boxed():
    result = extract_boxed_answer(r"First \boxed{1}, then \boxed{42}")
    assert result == "42"


def test_extract_no_boxed():
    assert extract_boxed_answer("No boxed answer here") is None


def test_verify_correct():
    assert verify_math_answer(r"Therefore \boxed{42}", "42") is True


def test_verify_wrong():
    assert verify_math_answer(r"Therefore \boxed{43}", "42") is False


def test_verify_equivalent_half():
    assert verify_math_answer(r"\boxed{0.5}", r"\frac{1}{2}") is True


def test_verify_equivalent_fraction():
    assert verify_math_answer(r"\boxed{\frac{3}{4}}", "0.75") is True


def test_verify_negative():
    assert verify_math_answer(r"\boxed{-5}", "-5") is True


def test_verify_no_boxed_in_prediction():
    # Should still try to extract a number
    result = verify_math_answer("The answer is 42", "42")
    assert result is True


def test_batch_verify():
    predictions = [r"\boxed{1}", r"\boxed{2}", r"\boxed{3}"]
    ground_truths = ["1", "99", "3"]
    results = batch_verify(predictions, ground_truths)
    assert results == [True, False, True]


def test_verify_integer_as_float():
    assert verify_math_answer(r"\boxed{6.0}", "6") is True


def test_verify_expression_equality():
    # 2^3 = 8
    assert verify_math_answer(r"\boxed{8}", "8") is True
