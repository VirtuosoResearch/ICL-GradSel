import torch
import numpy as np

def get_layers(model):
    """
    Utility function to get layers from the model.
    """
    layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            layers[name] = module
    return layers

def normalization(vs):
    """
    Normalizes a list of tensors (vectors) so that their concatenated norm is 1.
    """
    norm = torch.sqrt(sum([torch.sum(v**2) for v in vs]))
    return [v / (norm + 1e-6) for v in vs]

def compute_hessian_properties(model, loss, device="cpu", maxIter=100, tol=1e-8):
    """
    Computes the Hessian trace, maximal eigenvalue, and mean eigenvalue.
    Reuses Hessian-vector products to reduce computation cost.
    """
    # Move model and loss to the specified device
    model.to(device)
    loss = loss.to(device)

    # Ensure gradients are computed
    model.zero_grad()
    layers = get_layers(model)
    weights = [module.weight for name, module in layers.items()]
    weights = [w.to(device) for w in weights]

    # Compute gradients
    gradients = torch.autograd.grad(loss, weights, create_graph=True, retain_graph=True)

    # Total number of parameters
    total_params = sum(w.numel() for w in weights)

    # Initialize variables
    trace_vhv = []
    trace = 0.0
    eigenvalues = None
    vs_eigen = [torch.randn_like(w, device=device) for w in weights]
    vs_eigen = normalization(vs_eigen)
    Hv_eigen = None  # Will store the Hessian-vector product for eigenvalue computation
    reuse_Hv = False
    for iteration in range(maxIter):
        # Reuse Hv where possible

        # 1. Compute Hv for eigenvalue estimation
        if Hv_eigen is None or not reuse_Hv:
            Hv_eigen = torch.autograd.grad(gradients, weights, grad_outputs=vs_eigen, retain_graph=True)

        # Compute Rayleigh quotient (v^T H v) for eigenvalue estimation
        eigenvalue_estimate = sum([torch.sum(hv * v) for hv, v in zip(Hv_eigen, vs_eigen)]).item()

        # Normalize Hv to get next vs_eigen
        vs_eigen = normalization(Hv_eigen)

        # Check for convergence in eigenvalue estimation
        if eigenvalues is None:
            eigenvalues = eigenvalue_estimate
        else:
            if abs(eigenvalues - eigenvalue_estimate) / (abs(eigenvalues) + 1e-6) < tol:
                eigenvalues = eigenvalue_estimate
                # Since we have convergence in eigenvalue estimation, we can reuse Hv_eigen
                reuse_Hv = True
            else:
                eigenvalues = eigenvalue_estimate
                reuse_Hv = False  # Need to recompute Hv_eigen in the next iteration

        # 2. Compute Hv for trace estimation only if not reusing Hv_eigen
        if not reuse_Hv:
            # Generate Rademacher random vectors for trace estimation
            vs_trace = [torch.randint_like(w, high=2, device=device) for w in weights]
            vs_trace = [v.masked_fill(v == 0, -1) for v in vs_trace]  # Convert 0s to -1s

            # Compute Hv for trace estimation
            Hv_trace = torch.autograd.grad(gradients, weights, grad_outputs=vs_trace, retain_graph=True)

            # Compute v^T H v for trace estimation
            hvp_trace = sum([torch.sum(hv * v) for hv, v in zip(Hv_trace, vs_trace)]).item()
            trace_vhv.append(hvp_trace)

            # Check for convergence in trace estimation
            new_trace = np.mean(trace_vhv)
            if abs(new_trace - trace) / (abs(trace) + 1e-6) < tol and iteration > 0:
                trace = new_trace
                # If both trace and eigenvalue estimations have converged, break the loop
                if reuse_Hv:
                    break
            trace = new_trace
        else:
            # If Hv_eigen is being reused, use it for trace estimation
            hvp_trace = sum([torch.sum(hv * v) for hv, v in zip(Hv_eigen, vs_eigen)]).item()
            trace_vhv.append(hvp_trace)
            trace = np.mean(trace_vhv)
            # Since both estimations have converged, break the loop
            break

    hessian_trace = trace  # Estimated trace of the Hessian
    max_eigenvalue = eigenvalues  # Estimated maximal eigenvalue of the Hessian
    mean_eigenvalue = hessian_trace / total_params  # Mean eigenvalue

    return hessian_trace, max_eigenvalue, mean_eigenvalue