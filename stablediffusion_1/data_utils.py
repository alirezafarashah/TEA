from torch.utils.data import Dataset

from data_utils import (
    harmful_to_safe_prompts,
    vangogh_style_to_safe_prompts,
    kelly_mckernan_style_to_safe_prompts,
)


class PromptPairDatasetFromDict(Dataset):
    def __init__(self, prompt_dict):
        self.pairs = []
        for harmful, safe_list in prompt_dict.items():
            for safe in safe_list:
                self.pairs.append((harmful, safe))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
