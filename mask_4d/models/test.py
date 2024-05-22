import torch
from mask_4d.utils.instances import Tracks
import torch.nn as nn

query_feat = nn.Embedding(256, 100, device="cuda")
query_embed = nn.Embedding(256, 100, device="cuda")

def Init_Tracks():

    tracks = Tracks()
    N = query_feat.weight.shape[0]
    dev = query_feat.weight.device
    tracks.query = torch.empty((N, 256), device=dev)
    tracks.query_pe = torch.empty((N, 256), device=dev)
    tracks.id = torch.empty((N,), device=dev)
    tracks.life = torch.empty((N,), device=dev)
    tracks.center = torch.empty((N, 3), device=dev)
    tracks.size_xy = torch.empty((N, 3), device=dev)
    tracks.angle = torch.empty((N,), device=dev)
    tracks.bbox = torch.empty((N, 7), device=dev)
    return tracks


track_ins = Init_Tracks()
print(track_ins)
