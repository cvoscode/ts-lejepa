import types
import torch
import numpy as np
from src.timeseries.visualizations.callbacks import VisualizationCallback


class DummyExperiment:
    def __init__(self):
        self.figures = []
        self.images = []

    def add_figure(self, tag, fig, global_step=None):
        self.figures.append((tag, global_step))

    def add_image(self, tag, img, global_step=None):
        self.images.append((tag, img.shape, global_step))


class DummyLogger:
    def __init__(self):
        self.experiment = DummyExperiment()
        self.log_dir = None


class DummyTrainer:
    def __init__(self):
        self.loggers = [DummyLogger()]
        self.current_epoch = 0
        self.global_step = 0


def make_data(B=2, C=3, L_in=8, L_out=4):
    inputs = torch.randn(B, C, L_in)
    targets_bcl = torch.randn(B, C, L_out)
    targets_lcb = targets_bcl.transpose(1, 2).contiguous()
    preds_bcl = torch.randn(B, C, L_out)
    preds_lcb = preds_bcl.transpose(1, 2).contiguous()

    return inputs, targets_bcl, targets_lcb, preds_bcl, preds_lcb


def test_plot_predictions_bcl_shapes():
    inputs, targets_bcl, targets_lcb, preds_bcl, preds_lcb = make_data()
    cb = VisualizationCallback(num_samples_plot=1)
    trainer = DummyTrainer()

    data = {
        'input': inputs,
        'target': targets_bcl,
        'prediction': preds_bcl,
    }

    # Should not raise and should call add_figure
    cb._plot_predictions(trainer, data, step='UnitTest')
    assert len(trainer.loggers[0].experiment.figures) == 1
    tag, step_recorded = trainer.loggers[0].experiment.figures[0]
    assert 'UnitTest/Predictions' in tag


def test_plot_predictions_lcb_shapes():
    inputs, targets_bcl, targets_lcb, preds_bcl, preds_lcb = make_data()
    cb = VisualizationCallback(num_samples_plot=1)
    trainer = DummyTrainer()

    data = {
        'input': inputs,
        'target': targets_lcb,  # L,C,B layout
        'prediction': preds_lcb,
    }

    # Should not raise and should call add_figure
    cb._plot_predictions(trainer, data, step='UnitTest2')
    assert len(trainer.loggers[0].experiment.figures) == 1
    tag, step_recorded = trainer.loggers[0].experiment.figures[0]
    assert 'UnitTest2/Predictions' in tag


def test_plot_predictions_with_missing_target_or_pred():
    inputs, targets_bcl, targets_lcb, preds_bcl, preds_lcb = make_data()
    cb = VisualizationCallback(num_samples_plot=1)
    trainer = DummyTrainer()

    # Missing prediction
    data = {
        'input': inputs,
        'target': targets_bcl,
        'prediction': None,
    }
    cb._plot_predictions(trainer, data, step='NoPred')
    assert len(trainer.loggers[0].experiment.figures) == 1
    tag, _ = trainer.loggers[0].experiment.figures[0]
    assert 'NoPred/Predictions' in tag

    # Missing target
    trainer = DummyTrainer()
    data = {
        'input': inputs,
        'target': None,
        'prediction': preds_bcl,
    }
    cb._plot_predictions(trainer, data, step='NoTarget')
    assert len(trainer.loggers[0].experiment.figures) == 1
    tag, _ = trainer.loggers[0].experiment.figures[0]
    assert 'NoTarget/Predictions' in tag
