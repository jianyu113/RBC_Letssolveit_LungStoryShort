"""Small classifier training check using Vanessa's model and loaders."""
import copy
import hashlib
import json
import platform
from itertools import islice
from pathlib import Path

import pandas as pd
import torch
from torch import nn


def run_epoch(model, loader, device, criterion, max_batches, optimizer=None):
    """Train or validate a bounded number of batches; retain image identities."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    rows = []
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(islice(loader, max_batches), start=1):
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            if logits.shape != targets.shape:
                raise ValueError('Logits and labels must both have shape [B].')
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss; stop and inspect this batch.')
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item() * len(targets)
            scores = logits.detach().sigmoid().cpu().tolist()
            for image_id, patient_id, target, score in zip(
                batch['image_id'], batch['patient_id'], targets.cpu().tolist(), scores
            ):
                rows.append(dict(image_id=image_id, patient_id=patient_id,
                                 target=int(target), probability=score))
            print(f"{'Train' if training else 'Validation'} batch {step}: "
                  f"loss={loss.detach().item():.4f}", flush=True)
    if not rows:
        raise ValueError('No batches were processed.')
    return total_loss / len(rows), rows


def run_small_check(model, train_loader, validation_loader, device, output_dir,
                    settings, train_batches=5, validation_batches=5,
                    learning_rate=1e-4):
    """Run one bounded trial epoch, save outputs, and verify checkpoint reload."""
    if train_batches < 1 or validation_batches < 1:
        raise ValueError('Batch limits must be positive.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_head = model.fc.weight.detach().clone()
    train_loss, train_rows = run_epoch(
        model, train_loader, device, criterion, train_batches, optimizer
    )
    weights_updated = not torch.equal(initial_head, model.fc.weight.detach())
    if not weights_updated:
        raise RuntimeError('The classification head did not update.')

    # Compare parameters and buffers, including BatchNorm statistics.
    before_validation = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
    val_loss, val_rows = run_epoch(
        model, validation_loader, device, criterion, validation_batches
    )
    unchanged = all(torch.equal(v, model.state_dict()[k].detach().cpu())
                    for k, v in before_validation.items())
    if not unchanged:
        raise RuntimeError('Validation changed model parameters or buffers.')

    config = dict(settings, learning_rate=learning_rate, epochs=1,
                  max_train_batches=train_batches,
                  max_validation_batches=validation_batches,
                  python=platform.python_version(), torch=str(torch.__version__),
                  device=str(device), score_type='sigmoid probability',
                  purpose='pipeline check, not final performance evaluation')
    checkpoint_path = output_dir / 'checkpoint.pt'
    torch.save(dict(model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), epoch=1,
                    settings=config), checkpoint_path)

    # Reload the saved state and compare predictions on the same validation batch.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = copy.deepcopy(model)
    restored.load_state_dict(checkpoint['model_state_dict'])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=learning_rate)
    restored_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    restored.eval()
    probe = next(iter(validation_loader))['image'].to(device)
    with torch.no_grad():
        expected = model(probe)
        actual = restored(probe)
    torch.testing.assert_close(actual, expected)
    max_difference = (actual - expected).abs().max().item()

    pd.DataFrame(val_rows).to_csv(output_dir / 'validation_predictions.csv', index=False)
    pd.DataFrame(train_rows)[['image_id', 'patient_id', 'target']].to_csv(
        output_dir / 'training_samples.csv', index=False
    )
    metrics = dict(train_loss=train_loss, validation_loss=val_loss,
                   training_images=len(train_rows), validation_images=len(val_rows),
                   weights_updated=weights_updated, validation_unchanged=unchanged,
                   checkpoint_reload_passed=True,
                   checkpoint_max_logit_difference=max_difference,
                   test_set_used=False)
    (output_dir / 'settings.json').write_text(json.dumps(config, indent=2))
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f'Saved outputs to {output_dir}')
    return metrics


# Executable training uses the same interface as the verified notebook.
import argparse
import random
import numpy as np
from data_loader import create_classification_loaders
from model import create_resnet18


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def capture_random_states(generator):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=dict(name=numpy_state[0], keys=numpy_state[1].tolist(),
                           position=numpy_state[2], has_gauss=numpy_state[3],
                           cached_gaussian=numpy_state[4]),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader_generator=generator.get_state())


def restore_random_states(states, generator):
    random.setstate(states['python'])
    state = states['numpy']
    np.random.set_state((state['name'], np.array(state['keys'], dtype=np.uint32),
                         state['position'], state['has_gauss'], state['cached_gaussian']))
    torch.set_rng_state(states['torch_cpu'])
    if torch.cuda.is_available() and states['torch_cuda']:
        torch.cuda.set_rng_state_all(states['torch_cuda'])
    generator.set_state(states['loader_generator'])


def fit(model, train_loader, val_loader, generator, device, output_dir,
        epochs, learning_rate, settings, resume=None):
    """Train full epochs; resume only from a completed epoch boundary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    completed_epoch, best_loss, history = 0, float('inf'), []
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        for key in ('labels_sha256', 'batch_size', 'workers', 'seed'):
            if checkpoint['settings'][key] != settings[key]:
                raise ValueError(f'Resume settings differ: {key}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        completed_epoch = checkpoint['completed_epoch']
        best_loss = checkpoint['best_validation_loss']
        history = checkpoint['history']
        # Restore after construction/loading, before creating the next iterator.
        restore_random_states(checkpoint['random_states'], generator)
        print(f'Resuming after epoch {completed_epoch}; optimizer learning rate '
              f"{optimizer.param_groups[0]['lr']}", flush=True)
    if epochs <= completed_epoch:
        raise ValueError('--epochs must exceed the checkpoint completed epoch.')
    config = dict(settings, learning_rate=optimizer.param_groups[0]['lr'], epochs=epochs,
                  resume=str(resume) if resume else None, device=str(device),
                  torch=str(torch.__version__), python=platform.python_version())
    (output_dir / 'settings.json').write_text(json.dumps(config, indent=2))
    for epoch in range(completed_epoch + 1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}', flush=True)
        train_loss, train_rows = run_epoch(model, train_loader, device, criterion, None, optimizer)
        val_loss, predictions = run_epoch(model, val_loader, device, criterion, None)
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        metrics = dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                       training_images=len(train_rows), validation_images=len(predictions),
                       best_validation_loss=best_loss, test_set_used=False)
        history.append(metrics)
        checkpoint = dict(model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(),
                          completed_epoch=epoch, best_validation_loss=best_loss,
                          random_states=capture_random_states(generator),
                          settings=config, history=history)
        # Replace only complete checkpoint files, so an interrupted write keeps the old one.
        pending = output_dir / 'last.pending.pt'
        torch.save(checkpoint, pending)
        pending.replace(output_dir / 'last.pt')
        pd.DataFrame(predictions).to_csv(output_dir / 'validation_predictions.csv', index=False)
        if improved:
            torch.save(checkpoint, output_dir / 'best.pt')
            pd.DataFrame(predictions).to_csv(output_dir / 'best_validation_predictions.csv', index=False)
        pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)
        (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on the prepared train/val splits.')
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10, help='Total epochs, including completed epochs.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--no-pretrained', action='store_true', help='Start from random weights.')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error('Epochs, batch size and learning rate must be positive; workers nonnegative.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        parser.error('Use a new output directory for a fresh run, or supply --resume.')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    train_loader, val_loader = create_classification_loaders(
        args.prepared_dir, args.batch_size, args.workers, args.seed)
    generator = train_loader.generator
    # Resumption loads all weights from the checkpoint and needs no network download.
    model = create_resnet18(pretrained=not args.no_pretrained and args.resume is None).to(device)
    settings = dict(prepared_dir=str(args.prepared_dir), batch_size=args.batch_size,
                    workers=args.workers, seed=args.seed, model='resnet18',
                    labels_sha256=hashlib.sha256((args.prepared_dir / 'labels.csv').read_bytes()).hexdigest())
    fit(model, train_loader, val_loader, generator, device, args.output_dir,
        args.epochs, args.learning_rate, settings, args.resume)


if __name__ == '__main__':
    main()
