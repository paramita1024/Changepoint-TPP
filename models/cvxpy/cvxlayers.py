import cvxpy as cp 
from   models.cvxpy.cvxpylayers.torch import CvxpyLayer
import torch
import matplotlib.pyplot as plt
import numpy as np

# class CvxLayerMulti():
#     def __init__(self, n, k, safe):
#         x = cp.Parameter((k,n))
#         a = cp.Variable((k,n))
#         constraints = [a >= 0, a <= 1, cp.sum(a, axis=1) == 1]
#         A = cp.cumsum(a, axis=1)
#         constraints.append
#         # loss = 0
#         # for p in range(k):
#         #     if p != k-1:
#         #         loss += cp.sum(cp.multiply(x[p], A[p] - A[p+1]))
#         #     else:
#         #         loss += cp.sum(cp.multiply(x[p], A[p]))
#         # obj = cp.Maximize(loss)
#         obj = cp.Maximize(cp.sum(cp.multiply(x, A)))
#         if k > 1:
#             obj = cp.Maximize(cp.sum(cp.multiply(x,A)) - cp.sum(cp.multiply(A[1:,:], x[:-1,:])))
#         else:
#             obj = cp.Maximize(cp.sum(cp.multiply(x,A)))
        
#         prob = cp.Problem(obj, constraints)
#         self.cvxpylayer = CvxpyLayer(prob, parameters=[x], variables=[a])
    
#     def out(self, value):
#         sol, = self.cvxpylayer(value)

#         plt.figure()
#         for i in range(sol.shape[0]):
#             plt.plot(sol[i].detach())
#         plt.show()

#         print(sol.max(), sol.min())

#         out = torch.cumsum(sol, axis=1)

#         plt.figure()
#         for i in range(out.shape[0]):
#             plt.plot(out[i].detach())
#         plt.show()

#         return out


class CvxLayerMulti():
    def __init__(self, n, k, safe):

        x = cp.Parameter((k,n))
        a = cp.Variable((k,n))
        loss = 0
        constraints = []
        
        constraints.append( a >= 0)
        constraints.append( a <= 1)
        constraints.append( a[:,0] == 0 )
        constraints.append( a[:,n-1] == 1 )
        constraints.append( a[:,:n-1] <= a[:,1:] )
        if k>1:
            constraints.append( a[:k-1] >= a[1:])

        for p in range(k):
            # for i in range(n-1):
                # constraints.append(a[p,i+1] >= a[p,i])
                # constraints.append(a[p,i] >= 0)
                # constraints.append(a[p,i] <= 1)
            # constraints.append(a[p,0] == 0)
            # constraints.append(a[p,n-1] == 1)
            if p != k-1:
                # for i in range(n):
                #     constraints.append(a[p,i] >= a[p+1,i])
                constraints.append(cp.sum(a[p] - a[p+1]) >= safe)
                # loss += (a[p] - a[p+1])@x[p]
                loss += (a[p]@x[p])
            else:
                loss += (a[p]@x[p])
        
        obj = cp.Maximize(loss)
        prob = cp.Problem(obj, constraints)
        self.cvxpylayer = CvxpyLayer(prob, parameters=[x], variables=[a])
        
    def out(self, value):
        out, = self.cvxpylayer(value)
        return out

