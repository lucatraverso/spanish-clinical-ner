from huggingface_hub import hf_hub_download

REPO = "PlanTL-GOB-ES/pharmaconer"
FILES = {"train": "train-set_1.1.conll",
         "dev": "dev-set_1.1.conll",
         "test": "test-set_1.1.conll"}

def get_paths():
    return {split: hf_hub_download(REPO, f, repo_type="dataset")
            for split, f in FILES.items()}

def load_split(split):
    return read_conll(get_paths()[split])

def read_conll(path):
    sentences, tokens, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            cols = line.split()
            tokens.append(cols[0])
            tags.append(cols[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences