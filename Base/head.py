import torch
import torch.nn as nn

class Detect(nn.Module):
    # class Detect implements the head of object detection model
    # forces grid reconstruction
    dynamic = False
    # strides computed during build
    stride = None  
    # export mode
    export = False  

    def __init__(self, num_class=3, anchors=(), ch=(), inplace=True):
        super().__init__()
        # num of detected classes
        self.num_class = num_class
        
        # num of generated outputs (p, x, y, w, h, p1, p2, ..., pn)
        self.num_outputs = 5 + num_class

        # num of detection layers
        self.num_detect_layers = len(anchors)

        # num of anchors
        self.num_anchors = len(anchors[0]) // 2

        # init grid tensors
        self.grid = [torch.empty(0) for _ in range(self.num_detect_layers)]
        
        # init anchor_grid tensors
        self.grid_anchor = [torch.empty(0) for _ in range(self.num_detect_layers)]
        
        # save weight and biases as buffer(not trainable)
        # shape of the anchor tensor: [num_detect_layers, num_anchors, 2]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(
            self.num_detect_layers, -1, 2))
        
        # save modules in module list as output conv
        self.modelList = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in ch)
        
        self.inplace = inplace
        print("Detect constructor is finished!")

    def forward(self, x):
        # inference output
        inference = []
        for i in range(self.num_detect_layers):
            # select the output conv layer
            x[i] = self.modelList[i](x[i])
            
            # getting the shape sizes of the conv layer 
            bs, _, ny, nx = x[i].shape

            # change the shape of layer x from (bs,255,20,20) to (bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contigous()

            # inference mode
            if not self.training:  
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.grid_anchor[i] = self.make_grid(nx, ny, i)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_class + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.grid_anchor[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                inference.append(y.view(bs, self.num_anchors * nx * ny, self.num_outputs))

        return x if self.training else (torch.cat(inference, 1), ) if self.export else (torch.cat(inference, 1), x)

    def make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

if __name__ == "__main__":
    head = Detect(3, [[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]])