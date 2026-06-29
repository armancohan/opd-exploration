"""Math answer verification utilities."""

import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{} answer from text, handling nested braces."""
    pattern = r'\\boxed\{'
    starts = [m.start() for m in re.finditer(pattern, text)]
    if not starts:
        return None
    start = starts[-1] + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


def _try_sympy_parse(s: str):
    """Try to parse s as a sympy expression, return None on failure."""
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        return parse_expr(s, transformations=transformations, evaluate=True)
    except Exception:
        pass
    try:
        return parse_expr(s, evaluate=True)
    except Exception:
        return None


def _latex_to_sympy(s: str):
    """Convert latex fraction/sqrt/etc to sympy expression."""
    # Handle \frac{a}{b}
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    # Handle \sqrt{a}
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    # Handle \sqrt[n]{a}
    s = re.sub(r'\\sqrt\[([^\]]+)\]\{([^{}]+)\}', r'(\2)**(1/(\1))', s)
    # Remove remaining latex commands
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer string for comparison."""
    if answer is None:
        return ""
    s = answer.strip()
    # Remove $ signs
    s = s.replace('$', '').strip()
    return s


def verify_math_answer(prediction: str, ground_truth: str) -> bool:
    """Check if prediction contains an answer matching ground_truth."""
    pred_raw = extract_boxed_answer(prediction)
    if pred_raw is None:
        # Try last number in prediction as fallback
        nums = re.findall(r'-?\d+\.?\d*', prediction)
        if not nums:
            return False
        pred_raw = nums[-1]

    pred = normalize_math_answer(pred_raw)
    gt = normalize_math_answer(ground_truth)

    # Direct string match
    if pred.strip() == gt.strip():
        return True

    # Try numeric comparison
    try:
        p_val = float(pred)
        g_val = float(gt)
        return abs(p_val - g_val) < 1e-6
    except (ValueError, TypeError):
        pass

    # Try sympy symbolic comparison
    p_expr = _try_sympy_parse(_latex_to_sympy(pred))
    g_expr = _try_sympy_parse(_latex_to_sympy(gt))
    if p_expr is not None and g_expr is not None:
        try:
            diff = sympy.simplify(p_expr - g_expr)
            if diff == 0:
                return True
            # numeric check
            p_num = complex(p_expr.evalf())
            g_num = complex(g_expr.evalf())
            if abs(p_num - g_num) < 1e-6:
                return True
        except Exception:
            pass

    # Try latex-converted comparison
    p_conv = _latex_to_sympy(pred)
    g_conv = _latex_to_sympy(gt)
    if p_conv.strip() == g_conv.strip():
        return True

    return False


def batch_verify(predictions: list[str], ground_truths: list[str]) -> list[bool]:
    """Verify a batch of predictions against ground truths."""
    return [verify_math_answer(p, g) for p, g in zip(predictions, ground_truths)]
