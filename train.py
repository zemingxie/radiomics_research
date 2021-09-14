import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
import NiftiGenerator
import numpy as np


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        gen = NiftiGenerator.generate_paired_chunks( dataloader, chunk_size=(64,64,1), batch_size=64 )
        for idx in tqdm(range(1000)):
            x, y = next(gen)
            x_new = np.zeros((64,1,64,64))
            for i in range(64):
                temp = x[i,:,:,:]
                temp1 = np.transpose(temp,(2,0,1))
                x_new[i,:,:,:] = temp1
            y_new = np.zeros((64,1,64,64))
            for i in range(64):
                temp = y[i,:,:,:]
                temp1 = np.transpose(temp,(2,0,1))
                y_new[i,:,:,:] = temp1
            inputs, targets = torch.tensor(x_new, dtype=torch.double).to(self.device), torch.tensor(y_new, dtype=torch.double).to(self.device)
            loss, y_pred = self.batch_update(inputs, targets)

            # update loss logs
            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
            loss_logs = {self.loss.__name__: loss_meter.mean}
            logs.update(loss_logs)

            # update metrics logs
            for metric_fn in self.metrics:
                metric_value = metric_fn(y_pred, targets).cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
            logs.update(metrics_logs)

            if self.verbose:
                s = self._format_logs(logs)
            
            if idx%100 == 0:
                print(s)
                
        
          
        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
