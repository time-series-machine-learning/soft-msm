import torch

def _softmin2(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Smooth minimum of two tensors.

    Parameters
    ----------
    a, b : torch.Tensor
        Candidate values, broadcast-compatible and of identical shape.
    gamma : float
        Smoothness parameter (> 0).

    Returns
    -------
    torch.Tensor
        Soft-min over (a, b) with the same shape as the inputs.
    """
    v = torch.stack((a, b), dim=0)
    return -gamma * torch.logsumexp(-v / gamma, dim=0)


def _softmin3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Smooth minimum of three tensors.

    Parameters
    ----------
    a, b, c : torch.Tensor
        Candidate values, broadcast-compatible and of identical shape.
    gamma : float
        Smoothness parameter (> 0).

    Returns
    -------
    torch.Tensor
        Soft-min over (a, b, c) with the same shape as the inputs.
    """
    v = torch.stack((a, b, c), dim=0)
    return -gamma * torch.logsumexp(-v / gamma, dim=0)

def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances between time series.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (n_cases, n_channels, n_timepoints_x).
    y : torch.Tensor
        Input tensor of shape (n_cases, n_channels, n_timepoints_y).

    Returns
    -------
    torch.Tensor
        Pairwise squared distances of shape (n_cases, n_timepoints_x, n_timepoints_y).
    """
    x2 = (x * x).sum(dim=1)
    y2 = (y * y).sum(dim=1)
    xy = torch.bmm(x.transpose(1, 2), y)
    return x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy
