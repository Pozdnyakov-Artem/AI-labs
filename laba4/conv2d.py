import numpy as np
from linear import kaiming


def im2col(x, kernel_size, stride=1, padding=0):
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    N, C, H, W = x.shape
    Kh, Kw = kernel_size
    Hout = (H - Kh) // stride + 1
    Wout = (W - Kw) // stride + 1

    col = np.zeros((N, C * Kh * Kw, Hout * Wout))

    for n in range(N):
        col_idx = 0
        for i in range(0, Hout):
            for j in range(0, Wout):
                patch = x[n, :, i * stride:i * stride + Kh, j * stride:j * stride + Kw]
                col[n, :, col_idx] = patch.flatten()
                col_idx += 1
    return col, Hout, Wout

# a = np.array([[[1,2,3,4],
#      [5,6,7,8],
#      [9,10,11,12],
#      [13,14,15,16]]])
#
# print(im2col(a,(3,3)))

def col2im(dX_col, input_shape, kernel_size, stride=1, pad=0):

    N, C_in, H, W = input_shape
    Kh, Kw = kernel_size
    Hout = (H - Kh + 2 * pad) // stride + 1
    Wout = (W - Kw + 2 * pad) // stride + 1

    dX = np.zeros((N, C_in, H + 2 * pad, W + 2 * pad))

    for n in range(N):
        col_idx = 0
        for i in range(Hout):
            for j in range(Wout):
                patch_grad = dX_col[n, :, col_idx].reshape(C_in, Kh, Kw)
                dX[n, :, i * stride:i * stride + Kh, j * stride:j * stride + Kw] += patch_grad
                col_idx += 1

    if pad > 0:
        dX = dX[:, :, pad:-pad, pad:-pad]
    return dX

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = kaiming(np.random.randn(out_channels, in_channels, kernel_size[0],kernel_size[1]), in_channels*kernel_size[0]*kernel_size[1])
        self.b = np.zeros(out_channels)
        self.bias = bias
        self.cache_col = None
        self.cache_shape = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.cache_shape = x.shape
        col, Hout, Wout = im2col(x, self.kernel_size, self.stride, self.padding)
        self.cache_col = col.copy()
        self.cache_hout, self.cache_wout = Hout, Wout

        N = x.shape[0]
        W_flat = self.W.reshape(self.out_channels, -1)

        out = np.einsum('oc,ncm->nom', W_flat, col)

        if self.bias:
            out += self.b.reshape(1, self.out_channels, 1)  # broadcast

        return out.reshape(N, self.out_channels, Hout, Wout)

    def backward(self, dout):
        N = self.cache_shape[0]
        W_flat = self.W.reshape(self.out_channels, -1)

        dout_flat = dout.reshape(N, self.out_channels, -1)
        col = self.cache_col

        self.dW[:] = sum(
            np.matmul(dout_flat[n], col[n].T)
            for n in range(N)
        ).reshape(self.W.shape)

        if self.bias:
            self.db[:] = np.sum(dout, axis=(0, 2, 3))

        dx_col = np.zeros((N, self.in_channels * self.kernel_size[0] * self.kernel_size[1],
                           self.cache_hout * self.cache_wout))
        for n in range(N):
            dx_col[n] = np.matmul(W_flat.T, dout_flat[n])

        dx = col2im(dx_col, self.cache_shape, self.kernel_size, self.stride, self.padding)
        return dx

    def parameters(self):
        yield self.W, self.dW
        yield self.b, self.db