import pickle
import torch
from datetime import datetime


#####
# Path to datasets
#####
dataset_dir = 'dataset'
models_dir = 'models'
statistics_dir = 'statistics'


def save_model(model, optimizer, model_name, seed, timestamp, fingerprint):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_name': model_name,
        'seed': seed,
        'timestamp': timestamp,
    }, f"{models_dir}/{fingerprint}.pt")


def load_model(model, optimizer, fingerprint, evaluate=True):
    state = torch.load(f"{models_dir}/{fingerprint}.pt")
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    model_name = state['model_name']
    seed = state['seed']

    # Depending on purpose of model loading we call either `eval()` or `train()`.
    if evaluate:
        model.eval()
    else:
        model.train()

    return model_name, seed


def write_statistics(statistics, model_name, seed, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    if model_name not in statistics:
        statistics[model_name] = {}
    if seed not in statistics[model_name]:
        statistics[model_name][seed] = {}
    statistics[model_name][seed][epoch] = {
        'Training loss': train_loss,
        'Training accuracy': train_accuracy,
        'Validation loss': val_loss,
        'Validation accuracy': val_accuracy,
    }


def save_statistics(statistics, fingerprint):
    with open(f"{statistics_dir}/{fingerprint}.pkl", 'wb') as outp:
        pickle.dump(statistics, outp, pickle.HIGHEST_PROTOCOL)


def load_statistics(fingerprint):
    with open(f"{statistics_dir}/{fingerprint}.pkl", 'rb') as inp:
        return pickle.load(inp)


def get_timestamp():
    now = datetime.now()
    fill = lambda x: str(x).zfill(2)
    return f"{now.year}{fill(now.month)}{fill(now.day)}-{fill(now.hour)}{fill(now.minute)}{fill(now.second)}"