"""Validation-only selection and a durable, fail-closed final test gate."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .constants import LABELS


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    for item in sorted(path.rglob('*')) if path.is_dir() else [path]:
        if item.is_file():
            if path.is_dir():
                h.update(item.relative_to(path).as_posix().encode())
            with item.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def analyze(frame, predicted, directory, prefix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = classification_report(frame.label, predicted, labels=LABELS,
                                   output_dict=True, zero_division=0)
    errors = frame[['text', 'label']].copy()
    errors['predicted'] = predicted
    errors['length'] = errors.text.str.len()
    errors['has_negation'] = errors.text.str.contains(r"\b(?:not|never|no)\b|n't", case=False)
    errors['correct'] = errors.label == errors.predicted
    errors.loc[~errors.correct].to_csv(directory / f'{prefix}_errors.csv', index=False)
    slices = {str(key): {'rows': len(group), 'accuracy': float(group.correct.mean())}
              for key, group in errors.groupby('has_negation')}
    save_json(directory / f'{prefix}_analysis.json', {
        'classification_report': report, 'labels': LABELS,
        'confusion_matrix': confusion_matrix(frame.label, predicted, labels=LABELS).tolist(),
        'negation_slices': slices,
    })
    return {'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall']}


def register(run, name, artifact, backend, metrics, data_path):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'selection.json').exists():
        raise ValueError('Selection is frozen; use a new experiment directory')
    save_json(run / f'candidate_{name}.json', {
        'name': name, 'artifact': str(Path(artifact).resolve()), 'backend': backend,
        'metrics': metrics, 'evaluation_split': 'validation',
        'dataset_sha256': digest(data_path), 'artifact_sha256': digest(artifact),
    })


def select_model(run):
    run = Path(run)
    candidates = [json.loads(p.read_text()) for p in sorted(run.glob('candidate_*.json'))]
    if not candidates or len({c['dataset_sha256'] for c in candidates}) != 1:
        raise ValueError('Candidates must exist and use the same dataset')
    if any(c['evaluation_split'] != 'validation' for c in candidates):
        raise ValueError('Selection requires validation metrics')
    winner = max(candidates, key=lambda c: c['metrics']['f1_macro'])
    with (run / 'selection.json').open('x', encoding='utf-8') as stream:
        json.dump(winner, stream, indent=2)
    return winner


def evaluate_test(data_path, run):
    from .training import load_dataset
    run = Path(run)
    if (run / 'test_consumed.json').exists():
        raise FileExistsError('Final test already consumed for this experiment')
    selected = json.loads((run / 'selection.json').read_text())
    if digest(data_path) != selected['dataset_sha256']:
        raise ValueError('Dataset changed after validation')
    if digest(selected['artifact']) != selected['artifact_sha256']:
        raise ValueError('Selected model artifact changed')
    frame = load_dataset(data_path)
    test = frame[frame.split == 'test']
    if selected['backend'] == 'classical':
        model = joblib.load(selected['artifact'])
        predict = model.predict
    else:
        from .transformer import predict_local
        def predict(texts):
            return predict_local(selected['artifact'], list(texts))
    # Exclusive creation prevents concurrent/repeated test access. A failed attempt
    # stays consumed: deleting this marker would invalidate the evaluation protocol.
    with (run / 'test_consumed.json').open('x', encoding='utf-8') as stream:
        json.dump({'selection': selected, 'status': 'started'}, stream, indent=2)
    metrics = analyze(test, np.asarray(predict(test.text)), run, 'test')
    result = {'model': selected['name'], 'evaluation_split': 'test', 'metrics': metrics}
    save_json(run / 'test_summary.json', result)
    return result
