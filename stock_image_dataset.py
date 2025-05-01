import torch
import numpy as np
from torch.utils.data import Dataset


class StockImageDataset(Dataset):
    def __init__(self, stock_price, symbols, price_list, indicator_list, height, width):
        self.stock_price = stock_price
        self.symbols = symbols
        self.price_list = price_list
        self.indicator_list = indicator_list
        self.height = height
        self.width = width
        images, targets = self.create_images()
        self.images = torch.from_numpy(images).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def create_images(self):
        # Define size of window based on image size
        window = self.height * self.width

        # Define number of channels
        c = len(self.price_list) + len(self.indicator_list)

        # Define number of windows
        n = len(self.stock_price[self.symbols[0]].values) - window

        # Images are shaped as (N, C, H, W)
        images = np.zeros((1))  # Need to vstack images
        targets = np.zeros((1))  # Need to vstack stock prices as targets
        for i in range(n):  # This has to be a for loop through all rolling windows
            # Batch of images
            batch_images = np.zeros((len(self.symbols), c, self.height, self.width))
            batch_targets = np.zeros((len(self.symbols)))
            for symbol_index, symbol in enumerate(self.symbols):
                # Prices Processing
                for j, channel in enumerate(self.price_list):
                    channel_values = channel[symbol].values.flatten()
                    rolling_window = channel_values[i:i+window].reshape(self.height, self.width)

                    initial_price = rolling_window[0, 0]
                    if initial_price == 0:
                        normalized_window = np.zeros_like(rolling_window)
                    else:
                        normalized_window = rolling_window / initial_price
                    batch_images[symbol_index, j, :, :] = normalized_window
                    #batch_images[0, 0, :, :] = normalized_window

                    # Save off last stock price
                    if j == 0:  # This captures the Adj Close last stock price
                        batch_targets[symbol_index] = channel_values[i+window]
                # Indicator Processing
                for h, channel in zip(range(len(self.price_list), len(self.price_list) + len(self.indicator_list)), self.indicator_list):
                    channel_values = channel[symbol].values.flatten()
                    rolling_window = channel_values[i:i+window].reshape(self.height, self.width)
                    batch_images[symbol_index, h] = rolling_window
            if i != 0:
                images = np.vstack((images, batch_images))
                targets = np.append(targets, batch_targets)
            else:
                images = batch_images
                targets = batch_targets
        return images, targets
