import numpy as np
from sympy import prime_decomp
import torch
import faiss
import torchvision.transforms as T
from tqdm import tqdm
from salad.vpr_model import VPRModel

def load_model(ckpt_path, device):
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
    )

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    model = model.eval()
    # model = model.to('cuda')
    
    return model

def get_descriptors(model, dataloader, device):
    descriptors = []
    to_pil = T.ToPILImage()
    device_type = torch.device(device).type
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"):
            for batch in tqdm(dataloader, 'Calculating descriptors...'):
                imgs = batch[0]
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)

def get_nearest(q_list, r_list, k=1):
    q_list = q_list.detach().cpu().numpy().astype(np.float32)
    r_list = r_list.detach().cpu().numpy().astype(np.float32)
    
    embed_size = r_list.shape[1]
    faiss_index = faiss.IndexFlatL2(embed_size)
    faiss_index.add(r_list)  # Ensure float32 for FAISS compatibility

    _, indices = faiss_index.search(q_list, k)
    # predictions = [(i, indices[i][0]) for i in range(len(indices))]
    
    return indices
