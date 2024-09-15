from logging import getLogger
from multiprocessing import Value

import torch
logger = getLogger()

class MaskCollator(object):
    def __init__(
        self,
        ratio=(0.4, 0.6),  # (min, max) ratio of the mask size to the image size
        input_size=(224, 224),
        patch_size=16,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.ratio = ratio
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes (for distributed training)
        
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        
        Ouptut:
            collated_batch_org: original batch
            collated_masks_enc: masks for encoder block, representing the area to be not masked
            collated_masks_pred: masks for predictor block, representing the area to be masked
        '''
        # enc block or pred block may indicate th context block or the target block
        B = len(batch)
        
        collated_batch_org = torch.utils.data.default_collate(batch)  # Collates original batch
        
        # For distributed training, each process uses different seed to generate masks
        seed = self.step()  # use the shared counter to generate seed
        g = torch.Generator()
        g.manual_seed(seed)
        ratio = self.ratio  
        ratio = ratio[0] + torch.rand(1, generator=g).item() * (ratio[1] - ratio[0])    # masked patch ratio
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - ratio))
        
        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):
            m = torch.randperm(num_patches)
            collated_masks_enc.append(m[:num_keep])   # masks_enc represents area to be not masked
            collated_masks_pred.append(m[num_keep:])  # masks_pred represents area to be masked
        
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)  # (B, V), V: num of unmasked patches
        collated_masks_pred = torch.stack(collated_masks_pred, dim=0)  # (B, M), M: num of masked patches
        
        return collated_batch_org, collated_masks_enc, collated_masks_pred