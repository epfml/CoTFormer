import torch

def rotate_half(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return x.view(*x.shape[:-2], -1)

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1.):
    #freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    #t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos().to(t) * scale) + (rotate_half(t) * freqs.sin().to(t) * scale)
    #return torch.cat((t_left, t, t_right), dim = -1)
    return t
