import torch

def angle_dot(a, b):
    dot_product = torch.dot(a, b)
    prod_of_norms = torch.linalg.norm(a) * torch.linalg.norm(b)
    return dot_product / prod_of_norms  