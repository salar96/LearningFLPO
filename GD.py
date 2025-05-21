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
):

    optimizer = torch.optim.Adam([F_base], lr=optim_stepsize)

    for i in range(iters):

        D_min_drones, _, _ = VRPNet_pass(
            vrp_net, F_base, S, E, method="BeamSearch", returnGrad=False
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

        # print data
        if allowPrint:
            print(f"iter: {i}\tFreeE: {freeEnergy:.4e}\tNorm gradient: {Norm_G:.3f}\tmean_D_min:{torch.mean(D_min_drones).detach().item():.3e}")

    return F_base, freeEnergy, G
