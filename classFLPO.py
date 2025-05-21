import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import pickle
from scipy.spatial.distance import cdist
from scipy.optimize import LinearConstraint
from scipy.special import *
import cvxpy as cp

class FLPO():

    def __init__(self, filename:str, plotFLPO:bool, disType:str, selfHop:bool):

        #now load the mat file from the path and get the locations
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        self.n = data['numNodes']
        self.f = data['numFacilities']
        self.nodeLocations = data['nodeLocations']
        self.destinationLocation = data['destinationLocation']
        self.facilityLocations = data['facilityLocations']
        self.scale = data['scale']
        self.distance = disType
        self.selfHop = selfHop
        self.stageHorizon = self.f+1
        self.bounds = [(0,self.scale)]*self.f*2
        self.MY_INF = 1e8*self.scale
        self.nPaths = self.return_n_paths()

        if plotFLPO == True:
            self.plotFLPO()


    def plotFLPO(self):
        plt.figure(figsize=(4,3))
        plt.scatter(self.nodeLocations[:,0],self.nodeLocations[:,1],marker='o', alpha=0.2, label='nodes: '+str(self.n))
        # for facs in facilityLocations:
        #     plt.scatter(facs[:,0],facs[:,1],marker='v')
        plt.scatter(self.facilityLocations[:,0], self.facilityLocations[:,1], marker='v', label='facilities: '+ str(self.f))
        # plt.scatter(self.external_facilityLocations[:,0], self.external_facilityLocations[:,1], marker='d', label='ext_facilities: '+ str(self.f))
        plt.scatter(self.destinationLocation[:,0],self.destinationLocation[:,1], marker='D', label='destination: '+ str(np.round(self.destinationLocation.squeeze(), 2)))
        plt.grid()
        plt.legend()
        # legend = ['nodes: '+ str(self.n), 'facilities: ' + str(self.f), 'destination: ' + str(np.round(self.destinationLocation.squeeze(), 2))]
        # plt.legend(legend)
        plt.show()
        pass

    def return_n_paths(self):
        m = self.f
        n = self.n
        K = self.stageHorizon
        nPaths_flip = [0]*K
        for i in range(K):
            if i == 0:
                nPaths_flip[i] = np.ones(m+1)
            elif i < K:
                nPaths_flip[i] = np.ones(m+1) * np.sum(nPaths_flip[i-1])
            elif i == K:
                nPaths_flip[i] = np.ones(n) * np.sum(nPaths_flip[i])

        return nPaths_flip[::-1]

    def returnStagewiseCost(self, facs):
        # distance matrices
        t0 = time.time()
        dist_f2f = cdist(facs, facs, self.distance) + (1-self.selfHop)*np.diag([self.MY_INF]*self.f)
        dist_fd2d = cdist(np.concatenate((facs, self.destinationLocation)),self.destinationLocation,self.distance)
        dist_d2f = np.array([self.MY_INF]*self.f).reshape(1,-1)
        dist_fd2f = np.concatenate((dist_f2f, dist_d2f),axis=0)
        dist_fd2fd = np.concatenate((dist_fd2f, dist_fd2d), axis=1)
        t1 = time.time()        
        dist_n2fd = cdist(self.nodeLocations,np.concatenate((facs, self.destinationLocation)),self.distance)
        t2 = time.time()
        
        D_s = [0]*(self.stageHorizon)
        for k in range(self.stageHorizon):
            if k == 0: # n to fd
                D_s[k] = dist_n2fd
            elif k == self.stageHorizon-1: # f to d
                D_s[k] = dist_fd2d
            else:
                D_s[k] = dist_fd2fd

        tf = time.time()

        computeTime_dict = {
            'computeTime_fd2fd':t1-t0,
            'computeTime_n2fd':t2-t1,
            'computeTime_total':tf-t0
        }

        return D_s, computeTime_dict


    def returnStagewiseGrad(self, facs):

        m, d = facs.shape
        n = self.n

        t0 = time.time()
        # diff matrices
        diff_f2f = facs[:,None] - facs[None]
        diff_f2d = facs[:,None] - self.destinationLocation[None]
        diff_n2f = self.nodeLocations[:,None] - facs[None]

        # gradients: f2f
        grad_f2f = np.zeros(shape=(m, m, m, d))
        grad_f2f[np.arange(m), :, np.arange(m), :] = 2 * diff_f2f
        grad_f2f[:, np.arange(m), np.arange(m), :] = -2 * diff_f2f

        # gradients: f2d
        grad_f2d = np.zeros(shape=(m, 1, m, d))
        grad_f2d[np.arange(m), :, np.arange(m), :] = 2 * diff_f2d

        # gradients: d2fd
        grad_d2fd = np.zeros(shape=(1, m+1, m, d))
        t1 = time.time()

        # gradients: f2fd
        grad_f2fd = np.concatenate((grad_f2f, grad_f2d), axis=1)
        t2 = time.time()

        # gradients: d2d
        grad_d2d = np.zeros((1,1,m,d))

        # gradients: fd2d
        grad_fd2d = np.concatenate((grad_f2d, grad_d2d), axis=0)

        # gradients: fd2fd
        grad_fd2fd = np.concatenate((grad_f2fd, grad_d2fd), axis=0)
        t3 = time.time()

        # gradients: n2f
        grad_n2f = np.zeros(shape=(n, m, m, d))
        grad_n2f[:, np.arange(m), np.arange(m), :] = -2 * diff_n2f

        # gradients: n2d
        grad_n2d = np.zeros(shape=(n, 1, m, d))

        # gradients: n2fd
        grad_n2fd = np.concatenate((grad_n2f, grad_n2d), axis=1)
        t4 = time.time()

        GD_s = [0]*(self.stageHorizon)
        for k in range(self.stageHorizon):
            if k == 0: # n to fd
                GD_s[k] = grad_n2fd
            elif k == self.stageHorizon-1: # f to d
                GD_s[k] = grad_fd2d
            else:
                GD_s[k] = grad_fd2fd
        t5 = time.time()

        computeTime = {
            't_n2fd_fd2fd_fd2d': t4-t0,
            't_loop': t5-t4
        }

        return GD_s, computeTime


    def return_min_dist(self, facs, k2go):
        
        t0 = time.time()

        dist_f2f = cdist(facs, facs, self.distance) + (1-self.selfHop)*np.diag([self.MY_INF]*self.f)
        dist_fd2d = cdist(np.concatenate((facs, self.destinationLocation)),self.destinationLocation,self.distance)
        dist_d2f = np.array([self.MY_INF]*self.f).reshape(1,-1)
        dist_fd2f = np.concatenate((dist_f2f, dist_d2f),axis=0)
        dist_fd2fd = np.concatenate((dist_fd2f, dist_fd2d), axis=1)
        
        t1 = time.time()
        
        dist_n2fd = cdist(self.nodeLocations,np.concatenate((facs, self.destinationLocation)),self.distance)
        
        t2 = time.time()
        
        K = self.stageHorizon-k2go
        D_flip = [0]*K
        minD2go_flip = [0]*K
        for k in range(K):
            if k == 0:
                D_flip[k] = dist_fd2d
                D1 = D_flip[k]
            elif k > 0 and k < self.stageHorizon-1:
                D_flip[k] = dist_fd2fd
                D1 = D_flip[k] + np.tile(np.transpose(minD2go_flip[k-1]), (D_flip[k].shape[0],1))
            elif k == self.stageHorizon-1:
                D_flip[k] = dist_n2fd
                D1 = D_flip[k] + np.tile(np.transpose(minD2go_flip[k-1]), (D_flip[k].shape[0],1))
            minD2go_flip[k] = D1.min(axis=1, keepdims=True)

        t3 = time.time()

        computeTime = t3-t0

        return D_flip[::-1], minD2go_flip[::-1], computeTime


    def backPropDP(self, D_s, beta, returnPb=False):

        t0 = time.time()

        K = self.stageHorizon
        Lambda_flip = [0]*K
        V_flip = [0]*K
        D_flip = D_s[::-1]
        p_flip = [0]*K

        for i in range(K):
            if i == 0:
                Lambda_flip[i] = D_flip[i]
                V_flip[i] = D_flip[i]
            else:
                Lambda_flip[i] = D_flip[i] + np.tile(np.transpose(V_flip[i-1]), (D_flip[i].shape[0],1))
                minLambda = Lambda_flip[i].min(axis=1,keepdims=True)
                V_flip[i] = -1/beta*np.log(np.exp(-beta*(Lambda_flip[i] - minLambda)).sum(axis=1,keepdims=True)) + minLambda
            if returnPb:
                p_flip[i] = np.exp(-beta*(Lambda_flip[i]-V_flip[i]))
        
        tf = time.time()

        freeEnergy = np.sum(V_flip[K-1])/self.n

        return Lambda_flip[::-1], V_flip[::-1], freeEnergy, tf-t0, p_flip[::-1]
    

    def backPropDP_grad(self, GD_s, P):

        K = self.stageHorizon
        GV_flip = [0]*K
        GD_flip = GD_s[::-1]
        P_flip = P[::-1]

        for i in range(K):
            if i == 0:
                GV_flip[i] = GD_flip[i]
            else:
                GV_flip[i] = np.sum(P_flip[i][:,:,None,None] * (GD_flip[i] + GV_flip[i-1].squeeze()),axis=1,keepdims=True)

        G_freeEnergy = GV_flip[i].squeeze().sum(axis=0)/self.n

        return GV_flip[::-1], G_freeEnergy


    def objective(self, params, beta):
        D_s, _ = self.returnStagewiseCost(params.reshape(-1,2))
        _, _, finalCost, _, _ = self.backPropDP(D_s, beta)
        return finalCost


    def optimize_D(self, init_guess, beta, method, method_options):
        t0 = time.time()
        result = minimize(self.objective, init_guess, args=(beta,), bounds=self.bounds, method=method, options=method_options)
        params = result.x
        cost_fun=result.fun
        computeTime = time.time() - t0
        return cost_fun, params, computeTime


    def get_uY(self, Y, beta, gamma, p):

        # define variables
        uY = cp.Variable(Y.shape)
        delta = cp.Variable(1)

        t0 = time.time()
        # compute F and Fdot for constraints
        D_s, _ = self.returnStagewiseCost(Y)
        GD_s, _ = self.returnStagewiseGrad(Y)
        _, _, F, _, P = self.backPropDP(D_s, beta, returnPb=True)
        _, GF = self.backPropDP_grad(GD_s, P)
        t1 = time.time()
        # print(t1-t0)

        Fdot = cp.trace(GF.T @ uY)

        # CLF constraint
        constraints = [
            Fdot <= -gamma * (F + 1/beta * np.log(self.nPaths[0][0])) + delta]

        # objective and problem
        objective = cp.Minimize(cp.sum_squares(uY) + p * delta**2)
        problem = cp.Problem(objective, constraints)

        # solve
        problem.solve()

        return uY.value, Fdot.value


    def optimizeD_CLF(self, Y0, beta, tf, dt_init, dt_min, dt_max, p, gamma, tol, allowPrint=False):

        Y_prev = Y0
        iter_count = 0
        t = 0.0

        while t < tf:

            dYdt, Fdot = self.get_uY(Y_prev, beta, gamma, p)

            # compute adaptive step-size
            if iter_count > 0:
                step_size_1 = np.sqrt(1 + theta_prev) * dt_prev
                grad_diff = np.linalg.norm(dYdt - dYdt_old) + 1e-6
                step_size_2 = np.linalg.norm(Y_prev - Y_old) / (2 * grad_diff)  
                dt = min(max(step_size_2, dt_min), dt_max)
            else:
                dt = dt_init

            # update state
            Y_next = Y_prev + dYdt * dt

            # Compute new theta_k
            if iter_count > 0:
                theta = dt / dt_prev
            else:
                theta = 1.0  # Initial theta value

            # Update variables for next iteration
            Y_old = Y_prev
            dYdt_old = dYdt
            dt_prev = dt
            theta_prev = theta
            Y_prev = Y_next
            t += dt
            iter_count += 1
            
            if abs(Fdot) <= tol:
                print(f'time:{t:.3e}\tdt:{dt:.3f}\tFdot_tol={abs(Fdot):.3e} <= {tol}\tF:{F:.3e}')
                break
            elif allowPrint:
                print(f'time:{t:.3e}\tdt:{dt:.3f}\tFdot:{Fdot:.6f}\tF:{F:.6f}')
        
        return Y_next, F, Fdot

    



