# test_components.py


def function_call():
    x = 1
    y = 2
    z = function_call_helper(x, y)
    return z


def function_call_helper(arg0, arg1):
    return arg0 + arg1


def assignments():
    a, b = 0, 0
    c = 2 * a + 1
    d = b - c + 2
    a = c + 3
    return a, b, c, d


def fn_with_globals():
    global global_a, global_b, global_c
    global_a = 10
    global_b = 20
    global_c = 30
    return global_a + global_b + global_c


def fn_with_inner_fn():
    def inner_fn():
        while True:
            pass


def repeated_identifier():
    x = 0
    x = x + 1
    x = (x + (x + x)) + x
    return x
