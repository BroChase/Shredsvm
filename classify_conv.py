import pandas as pd


def classify_convert(x):
    if 0 < x <= 60:
        return 1
    if 60 < x <= 120:
        return 2
    if 120 < x <= 180:
        return 3
    if 180 < x <= 240:
        return 4
    if 240 < x <= 300:
        return 5
    if x > 300:
        return 6
