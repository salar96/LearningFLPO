import torch
import GD


def anneal(
    F_base,
    S,
    E,
    vrp_net,
    lse_net,
    optim_iters,
    optim_step,
    beta_min,
    beta_max,
    beta_grow,
    D_max_range,
    tol,
    optimizer_name,
    allowPrint,
):

    beta = beta_min
    
    b_arr = []
    F_arr = []

    while beta <= beta_max:
        # optimize using gradient descent iterations

        if optimizer_name == "gradient_descent0":
            F_base, FreeEnergy, G = GD.GD_at_beta0(
                F_base,
                S,
                E,
                vrp_net,
                lse_net,
                optim_iters,
                optim_step,
                beta_min,
                beta,
                D_max_range=D_max_range,
                tol = tol,
                allowPrint=False,
            )

        elif optimizer_name == "gradient_descent1":
            F_base, FreeEnergy, G = GD.GD_at_beta1(
                F_base,
                S,
                E,
                vrp_net,
                lse_net,
                optim_iters,
                optim_step,
                beta_min,
                beta,
                D_max_range=D_max_range,
                tol = tol,
                allowPrint=False,
            )

        elif optimizer_name == "adam":
            F_base, FreeEnergy, G = GD.Adam_at_beta(
                F_base,
                S,
                E,
                vrp_net,
                lse_net,
                optim_iters,
                optim_step,
                beta_min,
                beta,
                D_max_range=D_max_range,
                tol = tol,
                allowPrint=False,
            )

        # store data
        b_arr.append(beta)
        F_arr.append(torch.mean(FreeEnergy).detach().item())
        # print data
        if allowPrint:
            print(
                f"beta: {beta:.4e}\tFreeE: {torch.mean(FreeEnergy):.4f}\tGrad: {torch.norm(G):.4e}"
            )
        # update beta
        beta = beta * beta_grow

    return F_base, b_arr, F_arr
