r"""Text prompt database for multimodal VRP-SAM.

Loads pascal_text_prompts.json and provides train/eval text sampling.
Decoupled from any specific class-name convention via an alias mechanism,
so this module works whether your dataloader uses 'diningtable' or
'dining table' style spellings.

Usage:
    from common.text_prompt_db import TextPromptDB
    db = TextPromptDB('data/pascal_text_prompts.json')
    text  = db.sample_train(batch['class_name'])         # training
    texts = db.get_eval_ensemble(batch['class_name'])    # eval
"""
import json
import random
from typing import List, Optional, Dict


# Common spelling variations between dataset conventions and natural English.
# The values are the canonical keys used in pascal_text_prompts.json.
DEFAULT_NAME_ALIASES = {
    # natural form -> json key
    'dining table': 'diningtable',
    'potted plant': 'pottedplant',
    'tv monitor':   'tvmonitor',
    'tv/monitor':   'tvmonitor',
    'tv':           'tvmonitor',
    'television':   'tvmonitor',
}


class TextPromptDB:
    def __init__(self,
                 json_path: str,
                 p_template: float = 0.5,
                 use_synonyms: bool = False,
                 syn_prob: float = 0.3,
                 name_aliases: Optional[Dict[str, str]] = None):
        r"""
        Args:
            json_path: path to pascal_text_prompts.json
            p_template: prob of sampling from `templates` (else `descriptions`)
            use_synonyms: occasionally replace class name with a synonym
            syn_prob: prob of synonym replacement when use_synonyms=True
            name_aliases: extra {input_name: json_key} mappings if your
                          dataloader uses a spelling not covered by defaults
        """
        with open(json_path, 'r') as f:
            self.db = json.load(f)
        self.classes = self.db['classes']
        self.negative_pool = self.db.get('negative_pool', {})
        self.p_template = p_template
        self.use_synonyms = use_synonyms
        self.syn_prob = syn_prob

        # Build alias map: every recognized input string -> canonical json key
        self._aliases = dict(DEFAULT_NAME_ALIASES)
        if name_aliases:
            self._aliases.update(name_aliases)
        # Direct keys also alias to themselves
        for k in self.classes:
            self._aliases.setdefault(k, k)

    def _resolve(self, class_name: str) -> str:
        r"""Map an input class name to its canonical json key."""
        if class_name in self._aliases:
            return self._aliases[class_name]
        raise KeyError(
            "Class '{}' not found in TextPromptDB. Known classes: {}. "
            "Pass a `name_aliases` dict to TextPromptDB(...) if your "
            "dataloader uses a different spelling.".format(
                class_name, sorted(self.classes.keys())
            )
        )

    def sample_train(self, class_name: str) -> str:
        r"""Randomly sample one text prompt for training."""
        key = self._resolve(class_name)
        entry = self.classes[key]
        if random.random() < self.p_template:
            text = random.choice(entry['templates'])
        else:
            text = random.choice(entry['descriptions'])
        if self.use_synonyms and entry.get('synonyms') \
                and random.random() < self.syn_prob:
            syn = random.choice(entry['synonyms'])
            # Try every spelling that maps to this canonical key.
            # Longer candidates ('dining table') first to avoid partial matches.
            candidates = sorted(
                {key} | {k for k, v in self._aliases.items() if v == key},
                key=len, reverse=True,
            )
            for cand in candidates:
                if cand in text:
                    text = text.replace(cand, syn, 1)
                    break
        return text

    def get_eval_ensemble(self, class_name: str) -> List[str]:
        r"""Return all templates + descriptions for test-time ensemble."""
        key = self._resolve(class_name)
        entry = self.classes[key]
        return entry['templates'] + entry['descriptions']

    def get_hard_negative_names(self, class_name: str, k: int = 2) -> List[str]:
        r"""Return up to k visually-confusable class names (for contrastive loss)."""
        key = self._resolve(class_name)
        if key not in self.negative_pool:
            return []
        return self.negative_pool[key][:k]

    def get_parts(self, class_name: str) -> List[str]:
        r"""Return part names for compositional prompting (Innovation B)."""
        key = self._resolve(class_name)
        return self.classes[key].get('parts', [])