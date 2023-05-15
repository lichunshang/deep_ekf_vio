def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s

def geo_loss(Ps, Gs, do_scale = True, gamma = 0.9):
    n = Ps.shape[0]
    for i in range(n):
        w = gamma ** (n - i - 1)
        
        if do_scale:
            s = fit_scale(Ps, Gs)
    return 

