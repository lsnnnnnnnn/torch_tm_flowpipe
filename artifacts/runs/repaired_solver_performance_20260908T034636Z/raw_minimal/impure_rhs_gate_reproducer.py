from torch_tm_flowpipe import TMVector
external_values=[1.,2.,3.]
def impure(x, u=None):
    del external_values[0]
    return TMVector([x[0]+x[1],x[0]-x[1]])
