"""Exercise real Trainer/backprop/checkpoint saving without network or large weights."""
import json

import pandas as pd
import pytest


def test_tiny_roberta_training(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import ByteLevelBPETokenizer
    from english_sentiment import transformer as module
    from english_sentiment.constants import LABELS
    torch.set_num_threads(1)
    base = tmp_path / "tiny"
    base.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["good bad neutral sample train validation test"], vocab_size=270,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(base))
    tokenizer = transformers.RobertaTokenizerFast(
        vocab_file=str(base / "vocab.json"), merges_file=str(base / "merges.txt"))
    tokenizer.save_pretrained(base)
    config = transformers.RobertaConfig(vocab_size=len(tokenizer), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=132, num_labels=3)
    transformers.RobertaForSequenceClassification(config).save_pretrained(base)
    monkeypatch.setattr(module, "DEFAULT_MODEL", str(base))
    rows = [{"text": f"{split} {label}", "label": label, "split": split}
            for split in ("train", "validation", "test") for label in LABELS]
    data = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    reports = tmp_path / "reports"
    metrics = module.finetune_transformer(data, tmp_path / "models", reports,
                                         epochs=1, batch_size=3)
    assert 0 <= metrics["f1_macro"] <= 1
    candidate = json.loads((reports / "candidate_roberta.json").read_text())
    assert candidate["evaluation_split"] == "validation"
    assert not list(reports.glob("test*"))
    assert len(module.predict_local(candidate["artifact"], ["good"])) == 1
