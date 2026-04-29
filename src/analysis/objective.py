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

def _kge(observed, predicted):
    corr = correlation(observed, predicted)
    var = variability(observed, predicted)
    bias = predicted.mean() / observed.mean()
    return 1 - ((corr - 1) ** 2 + (var - 1) ** 2 + (bias - 1) ** 2) ** 0.5

def r2(observed, predicted):
    return 1 - ((predicted - observed) ** 2).sum() / ((observed - observed.mean()) ** 2).sum()