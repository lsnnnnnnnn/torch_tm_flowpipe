from torch_tm_flowpipe import TMVector
def rhs(x, u):
    del u
    return TMVector([1.0+x[0]*(x[0]*x[1]-4.0),x[0]*(3.0-x[0]*x[1])])
