from core.preprocessor import loader
from core.config import config
import core.model as models
import torch

def train(hidden_layer_sizes:list, max_iter = 2000):
    train_samples, train_labels = loader.load_all_csv_as_ndarray(config.PATH.DATA_TRAIN_CSV)

    network = models.BpNNRegressor([train_samples.shape[1]] + hidden_layer_sizes + [1], hidden_activation=torch.nn.ReLU)
    model = models.Model(network)
    trainer = models.Trainer(model)

    trainer.train(
        samples=train_samples, 
        labels=train_labels, 
        max_iter=max_iter,
        target_loss=1e-3
        )
        
    return model


if __name__ == "__main__":
    train([35], 10000)