import torch
import numpy

def get_exact_cw_pm(N, beta_lambda_max, num_points=1000, device="cuda"):
    # 1. Create an arbitrarily dense continuous grid from -1 to 1
    m_grid = torch.linspace(-1.0, 1.0, num_points, dtype=torch.float32, device=device)
    
    # 2. Map the continuous 'm' back to an effective continuous 'k' (number of up spins)
    k_cont = N * (m_grid + 1.0) / 2.0
    
    # 3. Use the continuous properties of the Gamma function (lgamma) 
    # to compute the combinatorial entropy for fractional spins
    log_entropy = torch.lgamma(torch.tensor(N + 1.0, device=device)) \
                - torch.lgamma(k_cont + 1.0) \
                - torch.lgamma(N - k_cont + 1.0)
                
    # 4. The energy term remains exactly the same
    energy = N * (beta_lambda_max / 2.0) * (m_grid ** 2)
    
    # 5. Combine and normalize
    log_p_m = log_entropy + energy
    p_m = torch.nn.functional.softmax(log_p_m, dim=0)
    
    return m_grid, p_m

import torch
import math
from ptt.pre_sampler import IIRCM

def initialize_cw_pre_sampler(visibles,ebm, beta_lambda_max, num_points=1000, device="cuda", dtype=torch.float32):
    mesh,p_m = get_exact_cw_pm(visibles,beta_lambda_max,num_points=10000,device=device)
    # 5. Initialize the BBRCM PreSampler
    U = torch.ones((1,visibles)).to(device)/visibles**0.5
    mu = torch.arctanh(mesh).reshape(-1,1)
    pre_sampler = IIRCM(
        ebm=ebm,
        p_m=p_m,
        mu=mu,
        U=U,
        device=device,
        dtype=dtype
    )
    
    return pre_sampler