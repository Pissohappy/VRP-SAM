r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal_enriched_text import DatasetPASCAL
from data.coco import DatasetCOCO
# from data.coco2pascal import DatasetCOCO2PASCAL

# === ADDED === text prompt database + default_collate (with version fallback)
from common.text_prompt_db import TextPromptDB
try:
    from torch.utils.data import default_collate          # PyTorch >= 1.11
except ImportError:
    from torch.utils.data._utils.collate import default_collate
# ==============


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize,
                   # === ADDED === text prompt configuration
                   text_prompt_json=None,
                   text_p_template=0.5,
                   text_use_synonyms=False,
                   text_syn_prob=0.3):
                   # ==============

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            # 'coco2pascal': DatasetCOCO2PASCAL
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

        # === ADDED === build text database if json path provided
        if text_prompt_json:
            cls.text_db = TextPromptDB(
                text_prompt_json,
                p_template=text_p_template,
                use_synonyms=text_use_synonyms,
                syn_prob=text_syn_prob,
            )
        else:
            cls.text_db = None
        # ==============

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        # === ADDED === conditionally inject text_db
        # (currently only pascal accepts text_db; coco can be added later)
        ds_kwargs = dict(
            datapath=cls.datapath, fold=fold, transform=cls.transform,
            split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize,
        )
        if benchmark == 'pascal' and cls.text_db is not None:
            ds_kwargs['text_db'] = cls.text_db
        dataset = cls.datasets[benchmark](**ds_kwargs)
        # ==============

        if split == 'trn':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=shuffle)
            shuffle = False
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=shuffle)
            pin_memory = True

        # === ADDED === custom collate_fn to keep text_prompt in [B, K] shape
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, pin_memory=True,
                                num_workers=nworker, sampler=sampler,
                                collate_fn=cls._collate_fn)
        # ==============

        return dataloader

    # === ADDED === custom collate
    @staticmethod
    def _collate_fn(batch):
        r"""Custom collate to handle text_prompt's variable type.

        In eval mode each sample's `text_prompt` is a List[str] (the ensemble).
        Default collate would transpose this to shape [K, B] (one inner list per
        ensemble position, containing one string per sample), which is unintuitive.
        We keep it as List[List[str]] of natural shape [B, K].

        All other fields (including `class_name`, `query_name`, `support_names`)
        fall through to default_collate, preserving original behavior.
        """
        extras = {}
        if 'text_prompt' in batch[0] and isinstance(batch[0]['text_prompt'], list):
            extras['text_prompt'] = [b['text_prompt'] for b in batch]
            batch = [{k: v for k, v in b.items() if k != 'text_prompt'} for b in batch]
        result = default_collate(batch)
        result.update(extras)
        return result
    # ==============