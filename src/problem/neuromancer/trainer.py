"""
Training pipeline
"""

import time

import copy
import torch

class trainer:
    def __init__(self, components, loss_fn, optimizer, epochs=100, growth_rate=1,
                 patience=5, warmup=0, clip=100, loss_key="loss", device="cpu"):
        """
        Initialize the Trainer class.
        """
        self.components = components
        self.loss_fn = loss_fn
        self.orig_weight = self.loss_fn.penalty_weight
        self.optimizer = optimizer
        self.epochs = epochs
        self.growth_rate = growth_rate
        self.patience = patience
        self.warmup = warmup
        self.clip = clip
        self.loss_key = loss_key
        self.device = device
        self.early_stop_counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None

    def train(self, loader_train, loader_dev):
        """
        Perform training with early stopping.
        """
        # init iter
        iters = 0
        stop_training = False
        # initial validation loss calculation
        self.components.eval()
        with torch.no_grad():
            val_loss = self.best_loss = self.calculate_loss(loader_dev)
        print(f"Epoch 0, Iters {iters}, Validation Loss: {val_loss:.2f}")
        # init accumulate training loss
        train_loss_total = 0
        # training loop
        tick = time.time()
        for epoch in range(self.epochs):
            # early stop
            if stop_training:
                break
            # go through data
            for data_dict in loader_train:
                # training phase
                self.components.train()
                # move to device
                for key in data_dict:
                    if torch.is_tensor(data_dict[key]):
                        data_dict[key] = data_dict[key].to(self.device)
                # forwad pass
                for comp in self.components:
                    data_dict.update(comp(data_dict))
                data_dict = self.loss_fn(data_dict)
                # backward pass
                data_dict[self.loss_key].backward()
                torch.nn.utils.clip_grad_norm_(self.components.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # accumulate train loss
                train_loss_total += data_dict[self.loss_key].item()
                iters += 1
                if iters % 125 == 0:
                    stop_training = self.validate(epoch, iters, loader_dev, train_loss_total/125)
                    # update penalty weight
                    self.loss_fn.penalty_weight *= self.growth_rate
                    # reset accumulated train loss
                    train_loss_total = 0
                    # early stop
                    if stop_training:
                        break
        tock = time.time()
        elapsed = tock - tick
        print("Training complete.")
        print(f"The training time is {elapsed:.2f} sec.")

    def validate(self, epoch, iters, loader_dev, train_loss):
        """
        validation
        """
        # validation phase
        self.components.eval()
        with torch.no_grad():
            # use orignal penalty weight for validation
            #self.loss_fn.penalty_weight, temp_weight = self.orig_weight, self.loss_fn.penalty_weight
            # get loss
            val_loss = self.calculate_loss(loader_dev)
            print(f"Epoch {epoch}, Iters {iters}, Training Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}")
            # restore weight
            #self.loss_fn.penalty_weight = temp_weight
        # turn into training phase
        self.components.train()
        # start early stop after warmup
        if iters // 125 >= self.warmup:
            # early stopping update
            self.update_early_stopping(val_loss)
            # check patience condition
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at iters {iters}")
                # load best model
                if self.best_model_state is not None:
                    self.components.load_state_dict(self.best_model_state)
                    print("Best model loaded.")
                return True
            else:
                return False

    def calculate_loss(self, loader):
        """
        Calculate loss for a given dataset loader.
        """
        total_loss = 0.0
        for data_dict in loader:
            # move to device
            for key in data_dict:
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key].to(self.device)
            # forward pass
            for comp in self.components:
                data_dict.update(comp(data_dict))
            total_loss += self.loss_fn(data_dict)[self.loss_key].item()
        return total_loss / len(loader)

    def update_early_stopping(self, val_loss):
        """
        Update the early stopping counter and model state.
        """
        # update with better loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(self.components.state_dict())
            self.early_stop_counter = 0  # reset early stopping counter
        else:
            self.early_stop_counter += 1
