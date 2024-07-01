import torch
import numpy as np

def transformer(source, target, out_size):
    """
    Thin Plate Spline Spatial Transformer Layer
    convert the TPS deformation into optical flows 
  TPS control points are arranged in arbitrary positions given by `source`.
  source : float Tensor [num_batch, num_point, 2] 
    The source position of the control points.
  target : float Tensor [num_batch, num_point, 2]
    The target position of the control points.
  out_size: tuple of two integers [height, width]
    The size of the output of the network (height, width)
    """

    def _meshgrid(height, width, source):

        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width]))
        if torch.cuda.is_available():
            x_t = x_t.cuda()
            y_t = y_t.cuda()
        
        x_t_flat = x_t.reshape([1, 1, -1])
        y_t_flat = y_t.reshape([1, 1, -1])

        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:,:,0], 2)
        py = torch.unsqueeze(source[:,:,1], 2)
        if torch.cuda.is_available():
            px = px.cuda()
            py = py.cuda()
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-9)
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)
        ones = torch.ones_like(x_t_flat_g)
        if torch.cuda.is_available():
            ones = ones.cuda()
        
        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)
        
        return grid

    def _transform(T, source, out_size):
        num_batch, *_ = T.size()

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, source)
        
        T_g = torch.matmul(T, grid)
        x_s = T_g[:,0,:]
        y_s = T_g[:,1,:]

        flow_x = (x_s - grid[:,1,:])*(out_width/2)
        flow_y = (y_s - grid[:,2,:])*(out_height/2)
            
        flow = torch.stack([flow_x, flow_y], 1)
        flow = flow.reshape([num_batch, 2, out_height, out_width])

        return flow


    def _solve_system(source, target):
        num_batch  = source.size()[0]
        num_point  = source.size()[1]
        
        np.set_printoptions(precision=8)
        
        ones = torch.ones(num_batch, num_point, 1).float()
        if torch.cuda.is_available():
            ones = ones.cuda()
        p = torch.cat([ones, source], 2)
        
        p_1 = p.reshape([num_batch, -1, 1, 3])
        p_2 = p.reshape([num_batch, 1, -1, 3])
        d2 = torch.sum(torch.square(p_1-p_2), 3)
        r = d2 * torch.log(d2 + 1e-9)
        
        zeros = torch.zeros(num_batch, 3, 3).float()
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        W_0 = torch.cat((p, r), 2)
        W_1 = torch.cat((zeros, p.permute(0,2,1)), 2)
        W = torch.cat((W_0, W_1), 1)
        W_inv = torch.inverse(W.type(torch.float64))
        
        zeros2 = torch.zeros(num_batch, 3, 2)
        if torch.cuda.is_available():
            zeros2 = zeros2.cuda()
        tp = torch.cat((target, zeros2), 1)
        T = torch.matmul(W_inv, tp.type(torch.float64))
        T = T.permute(0, 2, 1)

        return T.type(torch.float32)
  
    T = _solve_system(source, target)
    output = _transform(T, source, out_size)

    return output