import torch.nn as nn

''' Define 3D CNN model '''
class Model_3DCNN(nn.Module):

  def __conv_filter__(self, in_channels, out_channels, filter_size, stride, padding):
    conv_filter = nn.Sequential(nn.Conv3d(in_channels, out_channels, filter_size=filter_size, stride=stride, padding=padding, bias=True), nn.ReLU(inplace=True), nn.BatchNorm3d(out_channels))
    return conv_filter

  def __init__(self, feat_dim=19, output_dim=1, use_cuda=True):
    super(Model_3DCNN, self).__init__()     
    self.feat_dim = feat_dim
    self.output_dim = output_dim
    self.use_cuda = use_cuda
    
    # SE block
    self.conv_block1 = self.__conv_filter__(self.feat_dim, 64, 9, 2, 3)
    self.glob_pool1 = nn.AdaptiveAvgPool3d(1)
    self.SE_block = nn.Linear(in_features=64, out_features=64//16, bias=False)
    self.relu1 = nn.ReLU()
    self.SE_block1_ = nn.Linear(in_features=64//16, out_features=64, bias=False)
    self.sigmoid = nn.Sigmoid()

    # residual blocks
    self.res_block1 = self.__conv_filter__(64, 64, 7, 1, 3)
    self.res_block2 = self.__conv_filter__(64, 64, 7, 1, 3)

    # SE block
    self.conv_block2 = self.__conv_filter__(64, 128, 7, 3, 3)
    self.glob_pool = nn.AdaptiveAvgPool3d(1)
    self.SE_block2 = nn.Linear(in_features=128, out_features=128//16, bias=False)
    self.SE_block2_ = nn.Linear(in_features=128//16, out_features=128, bias=False)
    self.max_pool = nn.MaxPool3d(2)

    ## SE block
    self.conv_block3 = self.__conv_filter__(128, 256, 5, 2, 2)
    self.SE_block3 = nn.Linear(in_features=256, out_features=256//16, bias=False)
    self.SE_block3_ = nn.Linear(in_features=256//16, out_features=256, bias=False)

    # dense layers
    self.linear1 = nn.Linear(2048, 100)
    torch.nn.init.normal_(self.linear1.weight, 0, 1)
    self.linear1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1).train()
    self.linear2 = nn.Linear(100, 1)
    torch.nn.init.normal_(self.linear2.weight, 0, 1)

  def forward(self, x):
    if x.dim() == 1:
      x = x.unsqueeze(-1)

    # SE block 1
    conv1 = self.conv_block1(x)
    a1,b1, _, _, _ = conv1.shape
    glob_pool_conv1 = self.glob_pool(conv1).view(a1, b1)
    SE_block1 = self.SE_block1(glob_pool_conv1)   
    SE_block1a = self.relu(SE_block1)
    SE_block1_ = self.SE_block1_(SE_block1a)
    SE_block1_a = self.sigmoid(SE_block1_).view(a1, b1, 1, 1, 1)  
    se1 = conv1 * SE_block1_a.expand_as(conv1)  

    # residual blocks
    conv1_res1 = self.res_block1(se1)
    conv1_res12 = conv1_res1 + se1
    conv1_res2 = self.res_block2(conv1_res12)
    conv1_res2_2 = conv1_res2 + se1

    # SE block 2
    conv2 = self.conv_block2(conv1_res2_2)
    a2,b2, _, _, _ = conv2.shape
    glob_pool_conv2 = self.glob_pool(conv2).view(a2, b2)
    SE_block2 = self.SE_block2(glob_pool_conv2)        
    SE_block2a = self.relu(SE_block2)
    SE_block2_ = self.SE_block2_(SE_block2a)
    SE_block2_a = self.sigmoid(SE_block2_).view(a2, b2, 1, 1, 1)  
    se2 = conv2 * SE_block2_a.expand_as(conv2)  

    # Pooling layer
    pool2 = self.max_pool(se2)

    # SE block 3
    conv3 = self.conv_block3(pool2)
    a3,b3, _, _, _ = conv3.shape
    glob_pool_conv3 = self.glob_pool(conv3).view(a3, b3)
    SE_block3 = self.SE_block3(glob_pool_conv3)       
    SE_block3a = self.relu(SE_block3)
    SE_block3_ = self.SE_block3_(SE_block3a)
    SE_block3_a = self.sigmoid(SE_block3_).view(a3, b3, 1, 1, 1)  
    se3 = conv3 * SE_block3_a.expand_as(conv3)  

    # Pooling layer
    pool3 = se3

    # Flatten
    flatten = pool3.view(pool3.size(0), -1)

    # Linear layer 1
    linear1_z = self.linear1(flatten)
    linear1_y = self.relu(linear1_z)
    linear1 = self.linear1_bn(linear1_y) if linear1_y.shape[0]>1 else linear1_y

    # Linear layer 2
    linear2_z = self.linear2(linear1)
    
    return linear2_z, linear1_z
