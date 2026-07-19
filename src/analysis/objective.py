# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Objective functions used to score model predictions against observations."""


def rmse(observed, predicted):
    return ((predicted - observed) ** 2).mean() ** 0.5


def mae(observed, predicted):
    return (predicted - observed).abs().mean()


def bias(observed, predicted):
    return (predicted - observed).mean()


def variability(observed, predicted):
    return predicted.std() / observed.std()


def correlation(observed, predicted):
    return observed.corr(predicted)


def kge(observed, predicted):
    corr = correlation(observed, predicted)
    var = variability(observed, predicted)
    beta = predicted.mean() / observed.mean()
    return 1 - ((corr - 1) ** 2 + (var - 1) ** 2 + (beta - 1) ** 2) ** 0.5


def r2(observed, predicted):
    return 1 - ((predicted - observed) ** 2).sum() / ((observed - observed.mean()) ** 2).sum()


def all_metrics(observed, predicted) -> tuple:
    """Return (RMSE, MAE, bias, variability ratio, correlation, KGE, R squared)."""
    return (
        rmse(observed, predicted),
        mae(observed, predicted),
        bias(observed, predicted),
        variability(observed, predicted),
        correlation(observed, predicted),
        kge(observed, predicted),
        r2(observed, predicted),
    )
