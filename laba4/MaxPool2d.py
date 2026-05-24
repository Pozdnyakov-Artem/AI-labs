import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class MaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size[0]
        self.padding = padding
        self.cache = None

    def forward(self, x):
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        N, C, H, W = x.shape
        K_h, K_w = self.kernel_size
        H_out = (H - K_h) // self.stride + 1
        W_out = (W - K_w) // self.stride + 1

        patches = sliding_window_view(x, (K_h, K_w), axis=(-2, -1))
        patches = patches[:, :, ::self.stride, ::self.stride, :, :]

        patches_flat = patches.reshape(N, C, H_out, W_out, K_h * K_w)
        max_indices = np.argmax(patches_flat, axis=-1)

        out = np.max(patches, axis=(-2, -1))

        self.cache = (x.shape, max_indices, K_h, K_w)
        return out

    def backward(self, dout):
        input_shape, max_indices, K_h, K_w = self.cache
        N, C, H_in, W_in = input_shape
        N, C, H_out, W_out = max_indices.shape

        dx = np.zeros(input_shape)

        n_idx = np.arange(N)[:, None, None, None]
        c_idx = np.arange(C)[None, :, None, None]
        h_idx = np.arange(H_out)[None, None, :, None]
        w_idx = np.arange(W_out)[None, None, None, :]

        kh_offset = max_indices // K_w
        kw_offset = max_indices % K_w

        abs_h = h_idx * self.stride + kh_offset
        abs_w = w_idx * self.stride + kw_offset

        np.add.at(dx, (n_idx, c_idx, abs_h, abs_w), dout)

        if self.padding > 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dx