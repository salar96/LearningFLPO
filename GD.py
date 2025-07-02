import torch
from ModelPass import *


def gradient_descent_step(Y, dY, eta):
    return Y - eta * dY


# optimize using gradient descent at given beta
def GD_at_beta0(
    F_base,
    S,
    E,
    vrp_net,
    lse_net,
    iters,
    stepsize,
    beta_min,
    beta,
    D_max_range,
    tol = 1e-3,
    allowPrint=False,
):
    assert F_base.requires_grad == True

    for i in range(iters):

        # free energy and gradients using VRP_net and LSE_net
        D_min_drones, dDmin_dFlocs, _ = VRPNet_pass(
            vrp_net, F_base, S, E, method="Greedy", returnGrad=True
        )
        freeEnergy_drones, dFreeE_dDmin = LSENet_pass(
            lse_net,
            D_min_drones,
            D_max_range=D_max_range,
            beta=beta,
            beta_min=beta_min,
            returnGrad=True,
        )
        freeEnergy = torch.mean(freeEnergy_drones)

        torch.cuda.empty_cache()
        # print(dDmin_dFlocs.shape)
        # print(dFreeE_dDmin.shape)

        # total gradient using chain rule and backpropagation
        total_gradient = dDmin_dFlocs * dFreeE_dDmin
        G = torch.mean(total_gradient, axis=0)

        with torch.no_grad():
            Norm_G = torch.norm(G).item()
        if Norm_G < tol:
            if allowPrint:
                print(f'Optimization terminated due to tol at iteration: {i} FreeE: {freeEnergy:.4e}')
            break

        # optimizer step
        F_base = gradient_descent_step(F_base, G, stepsize)

        # print data
        if allowPrint:
            print(f"iter: {i}\tFreeE: {freeEnergy:.4e}\tNorm gradient: {Norm_G:.3f}\tmean_D_min:{torch.mean(D_min_drones).detach().item():.3e}")

    return F_base, freeEnergy_drones, G


# optimize using gradient descent at given beta
def GD_at_beta1(
    F_base,
    S,
    E,
    vrp_net,
    lse_net,
    iters,
    stepsize,
    beta_min,
    beta,
    D_max_range,
    tol = 1e-3,
    allowPrint=False,
):

    assert F_base.requires_grad == True

    for i in range(iters):

        # free energy and gradients using VRP_net and LSE_net
        D_min_drones, _, _ = VRPNet_pass(
            vrp_net, F_base, S, E, method="Greedy", returnGrad=False
        )

        freeEnergy_drones, _ = LSENet_pass(
            lse_net,
            D_min_drones,
            D_max_range=D_max_range,
            beta=beta,
            beta_min=beta_min,
            returnGrad=False,
        )
        freeEnergy = torch.mean(freeEnergy_drones)

        torch.cuda.empty_cache()

        # total gradient using chain rule and backpropagation
        total_gradient = torch.autograd.grad(
            outputs=freeEnergy,
            inputs=F_base,
            grad_outputs=torch.ones_like(freeEnergy),
            create_graph=True,
        )
        G = total_gradient[0]

        with torch.no_grad():
            Norm_G = torch.norm(G).item()
        if Norm_G < tol:
            if allowPrint:
                print(f'Optimization terminated due to tol at iteration: {i} FreeE: {freeEnergy:.4e}')
            break

        # optimizer step
        F_base = gradient_descent_step(F_base, G, stepsize)

        # print data
        if allowPrint:
            print(f"iter: {i}\tFreeE: {freeEnergy:.4e}\tNorm gradient: {Norm_G:.3f}\tmean_D_min:{torch.mean(D_min_drones).detach().item():.3e}")

    return F_base, freeEnergy_drones, G


def Adam_at_beta(
    F_base,
    S,
    E,
    vrp_net,
    lse_net,
    iters,
    optim_stepsize,
    beta_min,
    beta,
    D_max_range,
    tol = 1e-3,
    allowPrint=False,
    return_list = False
):

    optimizer = torch.optim.Adam([F_base], lr=optim_stepsize)
    if return_list:
        Y_arr = [F_base.clone().detach()]
    for i in range(iters):

        D_min_drones, _, _ = VRPNet_pass(
            vrp_net, F_base, S, E, method="sampling", returnGrad=False
        )
        freeEnergy_drones, _ = LSENet_pass(
            lse_net, D_min_drones, D_max_range=D_max_range, beta=beta, beta_min=beta_min
        )
        freeEnergy = torch.mean(freeEnergy_drones)

        torch.cuda.empty_cache()

        # compute gradient of free energy wrt F_base using backpropagation
        optimizer.zero_grad()
        freeEnergy.backward()
        G = F_base.grad
        with torch.no_grad():
            Norm_G = torch.norm(G).item()
        if Norm_G < tol:
            if allowPrint:
                print(f'Optimization terminated due to tol at iteration: {i} FreeE: {freeEnergy:.4e}')
            break
        # optimizer step
        optimizer.step()
        if return_list:
            Y_arr.append(F_base.clone().detach())
        # print data
        if allowPrint:
            print(f"iter: {i}\tFreeE: {freeEnergy:.4e}\tNorm gradient: {Norm_G:.3f}\tmean_D_min:{torch.mean(D_min_drones).detach().item():.3e}")
    if return_list:
        return Y_arr, freeEnergy, G
    else:
        return F_base, freeEnergy, G


def sampling_GD_at_beta(
    F_base,
    S,
    E,
    vrp_net,
    n_path_samples,
    beta,
    stepsize,
    iters,
    tol=1e-3,
    allowPrint=False
    ):
    assert F_base.requires_grad == True

    num_drones = S.shape[0]
    num_facilities = F_base.shape[1]
    dim_ = F_base.shape[2]
    print(f'n_drones:{num_drones}\tnum_facilities:{num_facilities}\tdim_:{dim_}')

    for i in range(iters):

        D_mins, GD_mins, _ = VRPNet_pass(
            vrp_net, 
            F_base, 
            S, 
            E,
            method="Greedy",
            returnGrad=True)

        D_samples, GD_samples = sampling_pass(
            F_base, 
            S, 
            E, 
            n_path_samples, 
            returnGrad=True)

        # compute a gibbs distribution on all the paths (shortest and sampled)
        D_cat = torch.cat((D_mins, D_samples.squeeze().T),axis=1)
        D_min = torch.min(D_cat, axis=1, keepdims=True).values
        D_off = D_cat - D_min
        D_exp = torch.exp(-beta * D_off)
        sumD_exp = torch.sum(D_exp, axis=1, keepdims=True)
        gibbs = D_exp/sumD_exp

        # compute net gradient using samples
        GD_mins_reshaped = GD_mins.reshape(1,num_drones,num_facilities,dim_)
        # print(GD_mins_reshaped.shape)
        # print(GD_samples.shape)
        Grads = torch.cat((GD_mins.reshape(1,num_drones,num_facilities,dim_), GD_samples), axis=0) # n_samples x n_drones x n_facilities x dim_
        reshape_gibbs = gibbs.T.reshape(n_path_samples+1, num_drones, 1, 1) # n_samples x n_drones x 1 x 1
        G = torch.sum(reshape_gibbs * Grads, axis=0) # resulting shape: n_drones x n_facilities x dim_

        # stopping criteria
        Norm_G = torch.norm(G).item()
        if Norm_G < tol:
            if allowPrint:
                print(f'Optimization terminated due to tol at iteration: {i} NormG: {NormG:.4e}')
            break

        # optimizer step
        F_base = gradient_descent_step(F_base, G, stepsize)

        # print data
        if allowPrint:
            print(f"iter: {i}\tNorm gradient: {Norm_G:.3f}\tmean_D_min:{torch.mean(D_mins).detach().item():.3e}")

    return F_base, G
