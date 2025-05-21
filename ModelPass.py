import time
import torch
import utils
from inference import inference


def VRPNet_pass(vrp_net, F_base, S, E, method="Greedy", returnGrad=False):
    num_drones = S.shape[0]
    F_locs = F_base.expand(num_drones, -1, -1)  # view, shares grad with F_base
    data = torch.cat((S, F_locs, E), dim=1)  # shape: (Nd, Nf+2, D)
    s = time.time()
    with torch.no_grad():
        _, actions = inference(data, vrp_net, method)  # Grad wrt actions not needed
    e = time.time()

    D_min_drones = utils.route_cost(data, actions).view(-1, 1)

    if returnGrad:
        # grad D_mins w.r.t. F_locs: shape = num_drones x num_facilities x dim_facility
        grad_outputs = torch.ones_like(D_min_drones)
        gradient = torch.autograd.grad(
            outputs=D_min_drones,
            inputs=F_locs,
            grad_outputs=grad_outputs,
        )

        dDmin_dFlocs = gradient[0]
        # dDmin_dFlocs = jacobian.squeeze().diag().view(-1,1)

    else:
        dDmin_dFlocs = None

    return D_min_drones, dDmin_dFlocs, e - s


# Free energy as a function of shortest path, and its gradients
def LSENet_pass(lse_net, D_min_drones, D_max_range, beta, beta_min, returnGrad=False):
    device = D_min_drones.device
    num_drones = D_min_drones.shape[0]
    Fmin_est = utils.area_approx_F(
        D_min_drones, D_max_range=D_max_range, beta=beta_min, printCalculations=False
    )
    Fmin_est.detach()
    In_lse = torch.cat(
        (
            Fmin_est.to(device),
            D_min_drones,
            (
                torch.ones(D_min_drones.shape, device=device)
                * torch.log(torch.tensor([beta], device=device))
                / torch.log(torch.tensor([10.0], device=device))
            ),
        ),
        axis=1,
    )
    FreeEnergy = lse_net(In_lse)

    if returnGrad:
        gradient = torch.autograd.grad(
            outputs=FreeEnergy,
            inputs=In_lse,
            grad_outputs=torch.ones_like(FreeEnergy),
        )
        # dFreeE_dDmin = gradient[0].view(num_drones,1,1)

        dFreeE_dDmin = gradient[0][:, 1].view(-1, 1).view(num_drones, 1, 1)
    else:
        dFreeE_dDmin = None
    return FreeEnergy, dFreeE_dDmin
