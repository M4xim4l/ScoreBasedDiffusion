import torch
import torch.nn as nn
from .models import AveragedModel
def score_loss(data, model, sigmas, device, enable_grad=True, noise_scale_idcs=None):
    if noise_scale_idcs is None:
        noise_scale_idcs = torch.randint(0, len(sigmas), (len(data),), dtype=torch.long, device=device)

    noise_sigmas = sigmas[noise_scale_idcs].view(data.shape[0], *([1] * len(data.shape[1:])))
    randn_noise = torch.randn_like(data)
    perturbation = noise_sigmas * randn_noise
    noisy_data = data + perturbation

    with torch.set_grad_enabled(enable_grad):
        output = model(noisy_data, noise_scale_idcs)
        regression_target = -1 / (noise_sigmas**2) * perturbation
        diff = (output - regression_target).view(len(data), -1)
        loss_expanded =  0.5 * (noise_sigmas**2).squeeze() * torch.sum(diff ** 2, dim=1)
        loss = torch.mean(loss_expanded, dim=0)

    return loss, loss_expanded


def train_score_model(model, sigmas, train_loader, lr, epochs, device,
                      optim='sgd', scheduler='cosine', weight_decay=0,
                      ema=False, val_loader=None):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplemented()

    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler == 'none':
        scheduler = None
    else:
        raise NotImplemented()

    if ema:
        avg_model = AveragedModel(model, device=device)
    else:
        avg_model = None

    N_noise_scales = len(sigmas)

    #generate a set of random idcs so we test every epoch at the same scales
    if val_loader is not None:
        val_random_idcs = torch.randint(0, N_noise_scales, (len(val_loader.dataset),), dtype=torch.long, device=device)
    else:
        val_random_idcs = None

    for epoch in range(epochs):
        avg_loss = 0

        model.train()
        for data, _ in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            loss, loss_expanded = score_loss(data, model, sigmas, device, noise_scale_idcs=None)
            loss.backward()
            avg_loss += loss / len(train_loader)
            optimizer.step()

            if ema:
                avg_model.update_parameters(model)


        print(f'Epoch {epoch} - Avg train loss {avg_loss}')
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            if ema:
                eval_model = avg_model
            else:
                eval_model = model
            eval_model.eval()
            avg_loss = 0
            with torch.no_grad():
                idx = 0
                for data, _ in val_loader:
                    data = data.to(device)

                    idx_next = idx + len(data)
                    noise_scale_idcs = val_random_idcs[idx:idx_next]
                    loss, loss_expanded = score_loss(data, eval_model, sigmas, device,
                                                     enable_grad=False, noise_scale_idcs=noise_scale_idcs)
                    avg_loss += torch.sum(loss_expanded, dim=0) / len(val_loader.dataset)
                    idx = idx_next

            print(f'Val loss {avg_loss}')

    if ema:
        return avg_model.module
    else:
        return model

#https://arxiv.org/pdf/2006.09011.pdf
#Technique 1
def estimate_max_distance_in_dataset(samples):

    # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist
    with torch.no_grad():
        D_mat = torch.sqrt(torch.clamp(pairwise_distances(samples), min=0))
        max_D = torch.max(D_mat.view(-1))
    return max_D
