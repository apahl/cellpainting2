#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as op
from ctypes import CDLL, POINTER, c_float, c_int, c_char_p


def main():
    lib_path = op.abspath(__file__)
    lib_path = op.realpath(lib_path)
    lib_path = op.dirname(lib_path)

    test_lib = CDLL(op.join(lib_path, "libctypes_ex.so"))

    # Function parameter types
    test_lib.joinList.argtypes = [POINTER(c_float), c_int]

    # Function return types
    test_lib.joinList.restype = c_char_p

    # Calc some numbers
    nums = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    nums_arr = (c_float * len(nums))()
    for i, v in enumerate(nums):
        nums_arr[i] = c_float(v)

    res = test_lib.joinList(nums_arr, c_int(len(nums_arr)))
    print('The len of %s is: %s' % (nums, res))


if __name__ == "__main__":
    main()
