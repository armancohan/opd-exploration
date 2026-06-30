"""Math answer verification utilities.

Primary backend is HuggingFace `math_verify` (robust LaTeX-aware grading used by
Open-R1). The hand-rolled sympy heuristic is kept as a fallback for environments
where `math_verify` is unavailable. Public API is unchanged:
`verify_math_answer`, `batch_verify`, `extract_boxed_answer`.
"""

import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ---------------------------------------------------------------------------
# Preferred backend: math_verify
# ---------------------------------------------------------------------------
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover - depends on install
    _HAS_MATH_VERIFY = False


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


def _mv_extract(s: str):
    """Parse a string with math_verify, forcing LaTeX mode for bare answers.

    Strings that already carry \\boxed{} or $...$ delimiters are parsed as-is so
    math_verify's boxed/dollar extraction kicks in. Bare answer strings (e.g.
    "x^2 + 1", "(3, \\frac{\\pi}{2})") are wrapped in $...$ so they are parsed as
    LaTeX expressions rather than falling through to last-number extraction.
    """
    if s is None:
        return None
    has_delim = ('\\boxed' in s) or ('$' in s)
    text = s if has_delim else f'${s}$'
    try:
        return _mv_parse(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback backend: sympy heuristic (original implementation)
# ---------------------------------------------------------------------------
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


def _verify_heuristic(prediction: str, ground_truth: str) -> bool:
    """Original hand-rolled sympy verifier (fallback)."""
    pred_raw = extract_boxed_answer(prediction)
    if pred_raw is None:
        nums = re.findall(r'-?\d+\.?\d*', prediction)
        if not nums:
            return False
        pred_raw = nums[-1]

    # Ground truth may itself be a full worked solution ending in \boxed{}.
    gt_raw = extract_boxed_answer(ground_truth)
    if gt_raw is None:
        gt_raw = ground_truth

    pred = normalize_math_answer(pred_raw)
    gt = normalize_math_answer(gt_raw)

    if pred.strip() == gt.strip():
        return True

    try:
        p_val = float(pred)
        g_val = float(gt)
        return abs(p_val - g_val) < 1e-6
    except (ValueError, TypeError):
        pass

    p_expr = _try_sympy_parse(_latex_to_sympy(pred))
    g_expr = _try_sympy_parse(_latex_to_sympy(gt))
    if p_expr is not None and g_expr is not None:
        try:
            diff = sympy.simplify(p_expr - g_expr)
            if diff == 0:
                return True
            p_num = complex(p_expr.evalf())
            g_num = complex(g_expr.evalf())
            if abs(p_num - g_num) < 1e-6:
                return True
        except Exception:
            pass

    p_conv = _latex_to_sympy(pred)
    g_conv = _latex_to_sympy(gt)
    if p_conv.strip() == g_conv.strip():
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def verify_math_answer(prediction: str, ground_truth: str) -> bool:
    """Check if prediction's answer matches ground_truth.

    `prediction` is typically a full model generation containing \\boxed{}.
    `ground_truth` may be a bare answer or a full reference solution (also with
    \\boxed{}). Uses math_verify when available, else the sympy heuristic.
    """
    if _HAS_MATH_VERIFY:
        gold = _mv_extract(ground_truth)
        pred = _mv_extract(prediction)
        if gold is not None and pred is not None:
            try:
                # math_verify signature is verify(gold, target)
                if _mv_verify(gold, pred) is True:
                    return True
            except Exception:
                pass
        # Fall through to heuristic if math_verify parsed nothing useful.
    return _verify_heuristic(prediction, ground_truth)


def batch_verify(predictions: list[str], ground_truths: list[str]) -> list[bool]:
    """Verify a batch of predictions against ground truths."""
    return [verify_math_answer(p, g) for p, g in zip(predictions, ground_truths)]
