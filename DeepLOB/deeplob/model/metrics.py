# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 2023/11/27 10:05
#
# pylint: disable=no-member

"""The metrics of y and y_hat.
    - r2_score
    - corr_score

"""

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None, mode: str = "global", **kwargs):
    """ :math:`R^2` (coefficient of determination) regression score function.
    :math:`R^2 = 1 - SSR/SST`.
    Supports weighting and three calculation modes `global` | `daily_mid` | `daily_mean`.

    Best possible score is 1.0, and it can be negative (because the model can be
    arbitrarily worse, it need not actually be the square of a quantity R).

    :param y_true: the label
    :param y_pred: the prediction
        The y_ture and y_pred MUST have the same shape:
        - shape=(num_of_samples) for mode == "global"
        - shape=(num_of_days, samples_of_one_day) for mode == "daily_mid/_mean"
    :param weight: the weight array, should have the same shape as y_true & y_pred
    :param mode: the mode to compute r2, you have only two choice now
        - global (default): compute r2 of all sample TOGETHER
        - daily_mid: compute r2 of all sample of each day and select the MID value
        - daily_mean: compute r2 of all sample of each day and compute the MEAN value

    return:
        - R2

    """

    # ---- Step 1. Test the shape and set the axis ---- #
    assert y_true.shape == y_pred.shape, "y_true, y_pred should have the same shape !"
    if mode == "global":  # if "global", shape=(num_of_samples)
        op_axis = tuple(range(len(y_true.shape)))
    elif mode in ("daily_mid", "daily_mean"):  # if  "daily_mid", shape=(num_of_days, samples_of_one_day)
        tick_num = kwargs.get("tick_num")
        y_true = y_true.reshape(-1, tick_num)
        y_pred = y_pred.reshape(-1, tick_num)
        weight = weight.reshape(-1, tick_num)
        op_axis = 1
    else:
        raise TypeError(f"mode = `{mode}` is not allowed now ! You can only choose `global` or `daily_mid`.")

    # ---- Step 2. Test the weight shape & make the default weight ---- #
    if weight is None:  # Make the default weight, all be 1
        weight = np.ones_like(y_true)
    else:
        assert weight.shape == y_true.shape, f"weight should have the same shape as y_true&y_pred !"

    # ---- Step 3. compute the SSR & SSE ---- #
    # SSE = sum(weight * (y - y_hat)^2))
    # shape=() just a number when "global", shape=(num_of_days,) when "daily_mid/_mean"
    numerator = np.sum(weight * ((y_true - y_pred) ** 2), axis=op_axis, dtype=np.float32)
    # two types (have y_bar or not) SST
    # SST = sum(y^2)
    denominator = np.sum(weight * (y_true ** 2), axis=op_axis, dtype=np.float32)

    # ---- Step 4. compute r2 = 1 - SSR/SST
    # shape=() just a number when "global", shape=(num_of_days,) when "daily_mid"
    r2 = 1 - (numerator / (denominator + 1e-10))
    if mode == "global":  # "global" just return
        return r2
    elif mode == "daily_mid":  # "daily_mid" get the median
        return np.median(r2)
    elif mode == "daily_mean":  # "daily_mean" compute the mean
        return np.mean(r2)


def corr_score(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None, mode: str = "global", **kwargs):
    """ :math:`correlation`
    :math:`corr = E[(y_true - y_true_bar)(y_pred - y_pred_bar)] / (std(y_true)*std(y_pred))`
    here we multiply `n - 1` and get:
        corr = sum((y_true - y_true_bar)(y_pred - y_pred_bar)) /
                [sqrt(sum((y_true - y_true_bar) ** 2)) * sqrt(sum((y_pred - y_pred_bar) ** 2))]
     Supports weighting and three calculation modes `global` | `daily_mid` | `daily_mean`.

    The corr could be [-1.0, 1.0], the 0 means no corr, 1 means strong positive corr, -1 means strong negative corr.

    :param y_pred: the prediction
    :param y_true: the label
        - shape=(num_of_samples) for mode == "global"
        - shape=(num_of_days, samples_of_one_day) for mode == "daily_mid/_mean"
    :param weight: the weight of label
    :param mode: the mode to compute corr, you have only two choice now
        - global (default): compute corr of all sample TOGETHER
        - daily_mid: compute corr of all sample of each day and select the MID value
        - daily_mean: compute corr of all sample of each day and compute the MEAN value

    return:
        - corr_score

    """

    # ---- Step 1. Test the shape and set the axis ---- #
    assert y_true.shape == y_pred.shape, "y_true, y_pred should have the same shape !"
    if mode == "global":  # if "global", shape=(num_of_samples)
        op_axis = tuple(range(len(y_true.shape)))
    elif mode in ("daily_mid", "daily_mean"):  # if  "daily_mid", shape=(num_of_days, samples_of_one_day)
        tick_num = kwargs.get("tick_num")
        y_true = y_true.reshape(-1, tick_num)
        y_pred = y_pred.reshape(-1, tick_num)
        weight = weight.reshape(-1, tick_num)
        op_axis = 1
    else:
        raise TypeError(f"mode = `{mode}` is not allowed now ! You can only choose `global` or `daily_mid`.")

    # ---- Step 2. Test the weight shape & make the default weight ---- #
    if weight is None:  # Make the default weight, all be 1
        weight = np.ones_like(y_true)
    else:
        assert weight.shape == y_true.shape, f"weight should have the same shape as y_true&y_pred !"

    # ---- Step 3. Compute numerator & denominator ---- #
    # sum(weight * (y_true * y_pred))
    numerator = np.sum(weight * (y_true * y_pred), axis=op_axis, dtype=np.float32)
    # sqrt(sum((y_true) ** 2)) * sqrt(sum((y_pred) ** 2))
    sum_y_true_std = np.sqrt(np.sum(weight * (y_true ** 2), axis=op_axis, dtype=np.float32))
    sum_y_pred_std = np.sqrt(np.sum(weight * (y_pred ** 2), axis=op_axis, dtype=np.float32))
    denominator = sum_y_true_std * sum_y_pred_std

    # ---- Step 3. ---- #
    # shape=() just a number when "global", shape=(num_of_days,) when "daily_mid"
    corr = numerator / (denominator + 1e-10)
    if mode == "global":  # "global" just return
        return corr
    elif mode == "daily_mid":  # "daily_mid" get the median
        return np.median(corr)
    elif mode == "daily_mean":  # "daily_mean" compute the mean
        return np.mean(corr)


if __name__ == "__main__":
    print("# ---- global test ---- #")
    # y_true = np.array([1, 2, 3])
    # y_pred = np.array([1, 2, 3])
    # weight = np.array([1, 1, 0])
    # # r2 = r2_score(y_true=y_true, y_pred=y_pred, weight=weight, mode="global", have_y_bar=True)
    # # print("r2 = ", r2)
    # corr = corr_score(y_true=y_true, y_pred=y_pred, weight=weight, mode="daily_mean", have_y_bar=True)
    # print("corr = ", corr)

    print("# ---- daily mid test ---- #")
    y_true = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y_pred = np.array([[-1, -2, -3], [2, 2, 2], [1, 2, 3]])
    # weight = np.array([[1, 1, 0], [1, 2, 0], [1, 2, 0]])
    r2 = r2_score(y_true=y_true, y_pred=y_pred, weight=None, mode="daily_mid", have_y_bar=True)
    print("r2 = ", r2)
    # corr = corr_score(y_true=y_true, y_pred=y_pred, mode="daily_mid")
    # print("corr = ", corr)
