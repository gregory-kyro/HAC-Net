import torch.nn as nn

'''Define a helper module for reshaping tensors'''
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

''' Define 3D-CNN model '''
class Model_3DCNN(nn.Module):

  def __conv_filter__(self, in_channels, out_channels, kernel_size, stride, padding):
    conv_filter = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True), nn.ReLU(inplace=True), nn.BatchNorm3d(out_channels))
    return conv_filter

  def __se_block__(self, channels):
    se_block = nn.Sequential(nn.AdaptiveAvgPool3d(1), View((-1,channels)), nn.Linear(channels, channels//16, bias=False), nn.ReLU(), 
                            nn.Linear(channels//16, channels, bias=False), nn.Sigmoid(), View((-1,channels,1,1,1)))
    return se_block

  def __init__(self, feat_dim=19, use_cuda=True):
    super(Model_3DCNN, self).__init__()     
    self.feat_dim = feat_dim
    self.use_cuda = use_cuda
    
    #Initialize model components
    self.conv_block1=self.__conv_filter__(self.feat_dim,64,9,2,3)
    self.se_block1=self.__se_block__(64)
    self.res_block1 = self.__conv_filter__(64, 64, 7, 1, 3)
    self.res_block2 = self.__conv_filter__(64, 64, 7, 1, 3)
    self.conv_block2=self.__conv_filter__(64, 128, 7, 3, 3)
    self.se_block2=self.__se_block__(128)
    self.max_pool = nn.MaxPool3d(2)
    self.conv_block3=self.__conv_filter__(128, 256, 5, 2, 2)
    self.se_block3=self.__se_block__(256)
    self.linear1 = nn.Linear(2048, 100)
    torch.nn.init.normal_(self.linear1.weight, 0, 1)
    self.relu=nn.ReLU()
    self.linear1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1).train()
    self.linear2 = nn.Linear(100, 1)
    torch.nn.init.normal_(self.linear2.weight, 0, 1)

  def forward(self, x):

    # SE block 1
    conv1 = self.conv_block1(x)
    squeeze1=self.se_block1(conv1)
    se1=conv1*squeeze1.expand_as(conv1) 

    # residual blocks
    conv1_res1 = self.res_block1(se1)
    conv1_res12 = conv1_res1 + se1
    conv1_res2 = self.res_block2(conv1_res12)
    conv1_res2_2 = conv1_res2 + se1

    # SE block 2
    conv2 = self.conv_block2(conv1_res2_2)
    squeeze2=self.se_block2(conv2)
    se2=conv2*squeeze2.expand_as(conv2) 

    # Pooling layer
    pool2 = self.max_pool(se2)

    # SE block 3
    conv3 = self.conv_block3(pool2)
    squeeze3=self.se_block3(conv3)
    se3=conv3*squeeze3.expand_as(conv3) 

    # Flatten
    flatten = se3.view(se3.size(0), -1)

    # Linear layer 1
    linear1_z = self.linear1(flatten)
    linear1_y = self.relu(linear1_z)
    linear1 = self.linear1_bn(linear1_y) if linear1_y.shape[0]>1 else linear1_y

    # Linear layer 2
    linear2_z = self.linear2(linear1)

    return linear2_z, flatten
