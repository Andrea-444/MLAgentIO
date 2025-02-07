def op_add(a: float, b: float) -> float:
    return a + b


def op_subtract(a: float, b: float) -> float:
    return a - b


def op_multiply(a: float, b: float) -> float:
    return a * b


def op_divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b