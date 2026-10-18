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
