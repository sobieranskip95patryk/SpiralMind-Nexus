from dataclasses import dataclass
from typing import Iterable, Iterator
import math

@dataclass
class BaseParams:
    W: int
    M: int
    D: int
    C: int
    A: int
    E: int
    T: int

def sum_of_digits(x: int) -> int:
    return sum(int(ch) for ch in str(abs(int(x))))

def reduce_to_9(x: int) -> int:
    if x == 0:
        return 0
    while x > 9:
        x = sum_of_digits(x)
    return x

def fib(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def s_base(params: BaseParams) -> int:
    sum5 = params.W + params.M + params.D + params.C + params.A
    raw = sum5 * params.E * params.T
    return reduce_to_9(raw)

def apply_formula_S(n: int, params: BaseParams) -> dict:
    S9 = s_base(params)
    S_pi = S9 * math.pi
    Fn = fib(n)
    return {"S9": S9, "S_pi": S_pi, "Fn": Fn, "WYNIK": S_pi + Fn}

def mix(value_logic: float, value_chaos: float, alpha: float) -> float:
    return (1.0 - alpha) * value_logic + alpha * value_chaos

def map_to_range(x: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return dst_min
    r = (x - src_min) / (src_max - src_min)
    return max(0.0, min(1.0, r)) * (dst_max - dst_min) + dst_min

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def event_source(seq: Iterable[str]) -> Iterator[str]:
    for item in seq:
        yield item

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    from collections import Counter
    cnt = Counter(text)
    n = sum(cnt.values())
    return -sum((c/n) * math.log2(c/n) for c in cnt.values())
