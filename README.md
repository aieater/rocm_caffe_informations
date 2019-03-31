# AMD RadeonGPU ROCm-Caffe information
<br>
<br>
<br>
This README is intended to provide helpful information for Deep Learning developers with AMD ROCm.<br>
<br>
Unfortunately, AMD's official repository for ROCm sometimes includes old or missing information. Therefore, on this readme, we will endeavor to describe accurate information based on the knowledge gained by GPUEater infrastructure development and operation.<br>
<br>
<br>
<br>
<br>

- How to setup Radeon GPU Driver (ROCm) on Ubuntu16.04/18.04
- How to setup ROCm-Caffe on Ubuntu16.04/18.04
  + ROCm(AMDGPU)-Caffe Python3.5 + UbuntuOS
<br>
<br>
<br>
<br>

### Python3.5 + ROCm Driver + ROCm-Caffe + easy installer (Recommend)
```
curl -sL http://install.aieater.com/setup_rocm_caffe | bash -
```

<br>
<br>

### Installation script of setup_rocm_caffe

```
sudo apt-get install -y \
     g++-multilib \
     libunwind-dev \
     git \
     cmake cmake-curses-gui \
     vim \
     emacs-nox \
     curl \
     wget \
     rpm \
     unzip \
     bc


sudo apt-get install -y rocm
sudo apt-get install -y rocm-libs
sudo apt-get install -y miopen-hip miopengemm


sudo apt-get install -y \
     pkg-config \
     protobuf-compiler \
     libprotobuf-dev \
     libleveldb-dev \
     libsnappy-dev \
     libhdf5-serial-dev \
     libatlas-base-dev \
     libboost-all-dev \
     libgflags-dev \
     libgoogle-glog-dev \
     liblmdb-dev \
     libfftw3-dev \
     libelf-dev

sudo pip3 install scikit-image scipy pyyaml protobuf

curl -sL http://install.aieater.com/setup_opencv | bash -


mkdir -p ~/src
cd ~/src
git clone https://github.com/ROCmSoftwarePlatform/hipCaffe.git
cd hipCaffe
cp ./Makefile.config.example ./Makefile.config
export USE_PKG_CONFIG=1
make -j$(nproc)
```


### Make sure to work

#### MNIST
```
cd ~/src/hipCaffe/
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
./examples/mnist/train_lenet.sh
```

#### CIFAR10
```
cd ~/src/hipCaffe/
./data/cifar10/get_cifar10.sh
./examples/cifar10/create_cifar10.sh
./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt
```



```
johndoe@thiguhag:~/src/hipCaffe$ ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt
I0331 11:06:32.843717 24302 caffe.cpp:217] Using GPUs 0
I0331 11:06:32.843881 24302 caffe.cpp:222] GPU 0: Vega 20
I0331 11:06:32.847487 24302 solver.cpp:48] Initializing solver from parameters: 
test_iter: 100
test_interval: 500
base_lr: 0.001
display: 100
max_iter: 4000
lr_policy: "fixed"
momentum: 0.9
weight_decay: 0.004
snapshot: 4000
snapshot_prefix: "examples/cifar10/cifar10_quick"
solver_mode: GPU
device_id: 0
net: "examples/cifar10/cifar10_quick_train_test.prototxt"
train_state {
  level: 0
  stage: ""
}
snapshot_format: HDF5
I0331 11:06:32.847564 24302 solver.cpp:91] Creating training net from net file: examples/cifar10/cifar10_quick_train_test.prototxt
I0331 11:06:32.847661 24302 net.cpp:322] The NetState phase (0) differed from the phase (1) specified by a rule in layer cifar
I0331 11:06:32.847671 24302 net.cpp:322] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy
I0331 11:06:32.847679 24302 net.cpp:58] Initializing net from parameters: 
name: "CIFAR10_quick"
state {
  phase: TRAIN
  level: 0
  stage: ""
}
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"
  }
  data_param {
    source: "examples/cifar10/cifar10_train_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.0001
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "pool1"
  top: "pool1"
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 64
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3"
  top: "pool3"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool3"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 64
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0331 11:06:32.847806 24302 layer_factory.hpp:77] Creating layer cifar
I0331 11:06:32.847857 24302 internal_thread.cpp:23] Starting internal thread on device 0
I0331 11:06:32.847921 24302 net.cpp:100] Creating Layer cifar
I0331 11:06:32.847929 24302 net.cpp:408] cifar -> data
I0331 11:06:32.847935 24302 net.cpp:408] cifar -> label
I0331 11:06:32.847940 24306 internal_thread.cpp:40] Started internal thread on device 0
I0331 11:06:32.847949 24302 data_transformer.cpp:25] Loading mean file from: examples/cifar10/mean.binaryproto
I0331 11:06:32.853466 24306 db_lmdb.cpp:35] Opened lmdb examples/cifar10/cifar10_train_lmdb
I0331 11:06:32.853821 24302 data_layer.cpp:41] output data size: 100,3,32,32
I0331 11:06:32.856909 24302 internal_thread.cpp:23] Starting internal thread on device 0
I0331 11:06:32.857230 24302 net.cpp:150] Setting up cifar
I0331 11:06:32.857236 24302 net.cpp:157] Top shape: 100 3 32 32 (307200)
I0331 11:06:32.857249 24302 net.cpp:157] Top shape: 100 (100)
I0331 11:06:32.857254 24302 net.cpp:165] Memory required for data: 1229200
I0331 11:06:32.857259 24302 layer_factory.hpp:77] Creating layer conv1
I0331 11:06:32.857255 24307 internal_thread.cpp:40] Started internal thread on device 0
I0331 11:06:32.857275 24302 net.cpp:100] Creating Layer conv1
I0331 11:06:32.857280 24302 net.cpp:434] conv1 <- data
I0331 11:06:32.857285 24302 net.cpp:408] conv1 -> conv1
I0331 11:06:33.909878 24302 net.cpp:150] Setting up conv1
I0331 11:06:33.909896 24302 net.cpp:157] Top shape: 100 32 32 32 (3276800)
I0331 11:06:33.909901 24302 net.cpp:165] Memory required for data: 14336400
I0331 11:06:33.909910 24302 layer_factory.hpp:77] Creating layer pool1
I0331 11:06:33.909919 24302 net.cpp:100] Creating Layer pool1
I0331 11:06:33.909924 24302 net.cpp:434] pool1 <- conv1
I0331 11:06:33.909932 24302 net.cpp:408] pool1 -> pool1
I0331 11:06:33.913565 24302 net.cpp:150] Setting up pool1
I0331 11:06:33.913594 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:33.913609 24302 net.cpp:165] Memory required for data: 17613200
I0331 11:06:33.913619 24302 layer_factory.hpp:77] Creating layer relu1
I0331 11:06:33.913635 24302 net.cpp:100] Creating Layer relu1
I0331 11:06:33.913653 24302 net.cpp:434] relu1 <- pool1
I0331 11:06:33.913669 24302 net.cpp:395] relu1 -> pool1 (in-place)
I0331 11:06:33.916409 24302 net.cpp:150] Setting up relu1
I0331 11:06:33.916424 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:33.916435 24302 net.cpp:165] Memory required for data: 20890000
I0331 11:06:33.916441 24302 layer_factory.hpp:77] Creating layer conv2
I0331 11:06:33.916460 24302 net.cpp:100] Creating Layer conv2
I0331 11:06:33.916467 24302 net.cpp:434] conv2 <- pool1
I0331 11:06:33.916482 24302 net.cpp:408] conv2 -> conv2
I0331 11:06:34.069100 24302 net.cpp:150] Setting up conv2
I0331 11:06:34.069113 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:34.069116 24302 net.cpp:165] Memory required for data: 24166800
I0331 11:06:34.069123 24302 layer_factory.hpp:77] Creating layer relu2
I0331 11:06:34.069130 24302 net.cpp:100] Creating Layer relu2
I0331 11:06:34.069133 24302 net.cpp:434] relu2 <- conv2
I0331 11:06:34.069137 24302 net.cpp:395] relu2 -> conv2 (in-place)
I0331 11:06:34.071858 24302 net.cpp:150] Setting up relu2
I0331 11:06:34.071878 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:34.071892 24302 net.cpp:165] Memory required for data: 27443600
I0331 11:06:34.071904 24302 layer_factory.hpp:77] Creating layer pool2
I0331 11:06:34.071923 24302 net.cpp:100] Creating Layer pool2
I0331 11:06:34.071934 24302 net.cpp:434] pool2 <- conv2
I0331 11:06:34.071949 24302 net.cpp:408] pool2 -> pool2
I0331 11:06:34.074851 24302 net.cpp:150] Setting up pool2
I0331 11:06:34.074873 24302 net.cpp:157] Top shape: 100 32 8 8 (204800)
I0331 11:06:34.074887 24302 net.cpp:165] Memory required for data: 28262800
I0331 11:06:34.074895 24302 layer_factory.hpp:77] Creating layer conv3
I0331 11:06:34.074916 24302 net.cpp:100] Creating Layer conv3
I0331 11:06:34.074928 24302 net.cpp:434] conv3 <- pool2
I0331 11:06:34.074944 24302 net.cpp:408] conv3 -> conv3
I0331 11:06:34.229825 24302 net.cpp:150] Setting up conv3
I0331 11:06:34.229837 24302 net.cpp:157] Top shape: 100 64 8 8 (409600)
I0331 11:06:34.229842 24302 net.cpp:165] Memory required for data: 29901200
I0331 11:06:34.229849 24302 layer_factory.hpp:77] Creating layer relu3
I0331 11:06:34.229856 24302 net.cpp:100] Creating Layer relu3
I0331 11:06:34.229859 24302 net.cpp:434] relu3 <- conv3
I0331 11:06:34.229863 24302 net.cpp:395] relu3 -> conv3 (in-place)
I0331 11:06:34.233310 24302 net.cpp:150] Setting up relu3
I0331 11:06:34.233340 24302 net.cpp:157] Top shape: 100 64 8 8 (409600)
I0331 11:06:34.233355 24302 net.cpp:165] Memory required for data: 31539600
I0331 11:06:34.233366 24302 layer_factory.hpp:77] Creating layer pool3
I0331 11:06:34.233382 24302 net.cpp:100] Creating Layer pool3
I0331 11:06:34.233397 24302 net.cpp:434] pool3 <- conv3
I0331 11:06:34.233412 24302 net.cpp:408] pool3 -> pool3
I0331 11:06:34.236271 24302 net.cpp:150] Setting up pool3
I0331 11:06:34.236287 24302 net.cpp:157] Top shape: 100 64 4 4 (102400)
I0331 11:06:34.236297 24302 net.cpp:165] Memory required for data: 31949200
I0331 11:06:34.236304 24302 layer_factory.hpp:77] Creating layer ip1
I0331 11:06:34.236325 24302 net.cpp:100] Creating Layer ip1
I0331 11:06:34.236336 24302 net.cpp:434] ip1 <- pool3
I0331 11:06:34.236348 24302 net.cpp:408] ip1 -> ip1
I0331 11:06:34.238878 24302 net.cpp:150] Setting up ip1
I0331 11:06:34.238896 24302 net.cpp:157] Top shape: 100 64 (6400)
I0331 11:06:34.238907 24302 net.cpp:165] Memory required for data: 31974800
I0331 11:06:34.238921 24302 layer_factory.hpp:77] Creating layer ip2
I0331 11:06:34.238935 24302 net.cpp:100] Creating Layer ip2
I0331 11:06:34.238945 24302 net.cpp:434] ip2 <- ip1
I0331 11:06:34.238955 24302 net.cpp:408] ip2 -> ip2
I0331 11:06:34.239763 24302 net.cpp:150] Setting up ip2
I0331 11:06:34.239779 24302 net.cpp:157] Top shape: 100 10 (1000)
I0331 11:06:34.239790 24302 net.cpp:165] Memory required for data: 31978800
I0331 11:06:34.239805 24302 layer_factory.hpp:77] Creating layer loss
I0331 11:06:34.239823 24302 net.cpp:100] Creating Layer loss
I0331 11:06:34.239831 24302 net.cpp:434] loss <- ip2
I0331 11:06:34.239841 24302 net.cpp:434] loss <- label
I0331 11:06:34.239851 24302 net.cpp:408] loss -> loss
I0331 11:06:34.239881 24302 layer_factory.hpp:77] Creating layer loss
I0331 11:06:34.242909 24302 net.cpp:150] Setting up loss
I0331 11:06:34.242923 24302 net.cpp:157] Top shape: (1)
I0331 11:06:34.242931 24302 net.cpp:160]     with loss weight 1
I0331 11:06:34.242945 24302 net.cpp:165] Memory required for data: 31978804
I0331 11:06:34.242951 24302 net.cpp:226] loss needs backward computation.
I0331 11:06:34.242960 24302 net.cpp:226] ip2 needs backward computation.
I0331 11:06:34.242966 24302 net.cpp:226] ip1 needs backward computation.
I0331 11:06:34.242972 24302 net.cpp:226] pool3 needs backward computation.
I0331 11:06:34.242978 24302 net.cpp:226] relu3 needs backward computation.
I0331 11:06:34.242985 24302 net.cpp:226] conv3 needs backward computation.
I0331 11:06:34.242990 24302 net.cpp:226] pool2 needs backward computation.
I0331 11:06:34.242997 24302 net.cpp:226] relu2 needs backward computation.
I0331 11:06:34.243002 24302 net.cpp:226] conv2 needs backward computation.
I0331 11:06:34.243010 24302 net.cpp:226] relu1 needs backward computation.
I0331 11:06:34.243016 24302 net.cpp:226] pool1 needs backward computation.
I0331 11:06:34.243022 24302 net.cpp:226] conv1 needs backward computation.
I0331 11:06:34.243029 24302 net.cpp:228] cifar does not need backward computation.
I0331 11:06:34.243036 24302 net.cpp:270] This network produces output loss
I0331 11:06:34.243049 24302 net.cpp:283] Network initialization done.
I0331 11:06:34.243284 24302 solver.cpp:181] Creating test net (#0) specified by net file: examples/cifar10/cifar10_quick_train_test.prototxt
I0331 11:06:34.243317 24302 net.cpp:322] The NetState phase (1) differed from the phase (0) specified by a rule in layer cifar
I0331 11:06:34.243336 24302 net.cpp:58] Initializing net from parameters: 
name: "CIFAR10_quick"
state {
  phase: TEST
}
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"
  }
  data_param {
    source: "examples/cifar10/cifar10_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.0001
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "pool1"
  top: "pool1"
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 64
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3"
  top: "pool3"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool3"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 64
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0331 11:06:34.243697 24302 layer_factory.hpp:77] Creating layer cifar
I0331 11:06:34.243782 24302 internal_thread.cpp:23] Starting internal thread on device 0
I0331 11:06:34.243839 24302 net.cpp:100] Creating Layer cifar
I0331 11:06:34.243852 24302 net.cpp:408] cifar -> data
I0331 11:06:34.243862 24302 net.cpp:408] cifar -> label
I0331 11:06:34.243872 24302 data_transformer.cpp:25] Loading mean file from: examples/cifar10/mean.binaryproto
I0331 11:06:34.243873 24322 internal_thread.cpp:40] Started internal thread on device 0
I0331 11:06:34.251590 24322 db_lmdb.cpp:35] Opened lmdb examples/cifar10/cifar10_test_lmdb
I0331 11:06:34.252020 24302 data_layer.cpp:41] output data size: 100,3,32,32
I0331 11:06:34.255237 24302 internal_thread.cpp:23] Starting internal thread on device 0
I0331 11:06:34.255578 24302 net.cpp:150] Setting up cifar
I0331 11:06:34.255586 24302 net.cpp:157] Top shape: 100 3 32 32 (307200)
I0331 11:06:34.255594 24302 net.cpp:157] Top shape: 100 (100)
I0331 11:06:34.255599 24302 net.cpp:165] Memory required for data: 1229200
I0331 11:06:34.255604 24302 layer_factory.hpp:77] Creating layer label_cifar_1_split
I0331 11:06:34.255617 24302 net.cpp:100] Creating Layer label_cifar_1_split
I0331 11:06:34.255622 24302 net.cpp:434] label_cifar_1_split <- label
I0331 11:06:34.255630 24302 net.cpp:408] label_cifar_1_split -> label_cifar_1_split_0
I0331 11:06:34.255632 24323 internal_thread.cpp:40] Started internal thread on device 0
I0331 11:06:34.255650 24302 net.cpp:408] label_cifar_1_split -> label_cifar_1_split_1
I0331 11:06:34.261567 24302 net.cpp:150] Setting up label_cifar_1_split
I0331 11:06:34.261586 24302 net.cpp:157] Top shape: 100 (100)
I0331 11:06:34.261595 24302 net.cpp:157] Top shape: 100 (100)
I0331 11:06:34.261620 24302 net.cpp:165] Memory required for data: 1230000
I0331 11:06:34.261626 24302 layer_factory.hpp:77] Creating layer conv1
I0331 11:06:34.261641 24302 net.cpp:100] Creating Layer conv1
I0331 11:06:34.261646 24302 net.cpp:434] conv1 <- data
I0331 11:06:34.261651 24302 net.cpp:408] conv1 -> conv1
I0331 11:06:34.405349 24302 net.cpp:150] Setting up conv1
I0331 11:06:34.405360 24302 net.cpp:157] Top shape: 100 32 32 32 (3276800)
I0331 11:06:34.405364 24302 net.cpp:165] Memory required for data: 14337200
I0331 11:06:34.405372 24302 layer_factory.hpp:77] Creating layer pool1
I0331 11:06:34.405380 24302 net.cpp:100] Creating Layer pool1
I0331 11:06:34.405382 24302 net.cpp:434] pool1 <- conv1
I0331 11:06:34.405386 24302 net.cpp:408] pool1 -> pool1
I0331 11:06:34.408463 24302 net.cpp:150] Setting up pool1
I0331 11:06:34.408470 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:34.408478 24302 net.cpp:165] Memory required for data: 17614000
I0331 11:06:34.408483 24302 layer_factory.hpp:77] Creating layer relu1
I0331 11:06:34.408491 24302 net.cpp:100] Creating Layer relu1
I0331 11:06:34.408496 24302 net.cpp:434] relu1 <- pool1
I0331 11:06:34.408502 24302 net.cpp:395] relu1 -> pool1 (in-place)
I0331 11:06:34.411069 24302 net.cpp:150] Setting up relu1
I0331 11:06:34.411075 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:34.411082 24302 net.cpp:165] Memory required for data: 20890800
I0331 11:06:34.411087 24302 layer_factory.hpp:77] Creating layer conv2
I0331 11:06:34.411098 24302 net.cpp:100] Creating Layer conv2
I0331 11:06:34.411103 24302 net.cpp:434] conv2 <- pool1
I0331 11:06:34.411110 24302 net.cpp:408] conv2 -> conv2
I0331 11:06:34.555790 24302 net.cpp:150] Setting up conv2
I0331 11:06:34.555802 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:34.555809 24302 net.cpp:165] Memory required for data: 24167600
I0331 11:06:34.555826 24302 layer_factory.hpp:77] Creating layer relu2
I0331 11:06:34.555835 24302 net.cpp:100] Creating Layer relu2
I0331 11:06:34.555841 24302 net.cpp:434] relu2 <- conv2
I0331 11:06:34.555847 24302 net.cpp:395] relu2 -> conv2 (in-place)
I0331 11:06:34.558418 24302 net.cpp:150] Setting up relu2
I0331 11:06:34.558424 24302 net.cpp:157] Top shape: 100 32 16 16 (819200)
I0331 11:06:34.558430 24302 net.cpp:165] Memory required for data: 27444400
I0331 11:06:34.558435 24302 layer_factory.hpp:77] Creating layer pool2
I0331 11:06:34.558442 24302 net.cpp:100] Creating Layer pool2
I0331 11:06:34.558447 24302 net.cpp:434] pool2 <- conv2
I0331 11:06:34.558454 24302 net.cpp:408] pool2 -> pool2
I0331 11:06:34.561081 24302 net.cpp:150] Setting up pool2
I0331 11:06:34.561089 24302 net.cpp:157] Top shape: 100 32 8 8 (204800)
I0331 11:06:34.561094 24302 net.cpp:165] Memory required for data: 28263600
I0331 11:06:34.561098 24302 layer_factory.hpp:77] Creating layer conv3
I0331 11:06:34.561111 24302 net.cpp:100] Creating Layer conv3
I0331 11:06:34.561116 24302 net.cpp:434] conv3 <- pool2
I0331 11:06:34.561122 24302 net.cpp:408] conv3 -> conv3
I0331 11:06:34.707540 24302 net.cpp:150] Setting up conv3
I0331 11:06:34.707552 24302 net.cpp:157] Top shape: 100 64 8 8 (409600)
I0331 11:06:34.707562 24302 net.cpp:165] Memory required for data: 29902000
I0331 11:06:34.707576 24302 layer_factory.hpp:77] Creating layer relu3
I0331 11:06:34.707585 24302 net.cpp:100] Creating Layer relu3
I0331 11:06:34.707590 24302 net.cpp:434] relu3 <- conv3
I0331 11:06:34.707597 24302 net.cpp:395] relu3 -> conv3 (in-place)
I0331 11:06:34.710366 24302 net.cpp:150] Setting up relu3
I0331 11:06:34.710372 24302 net.cpp:157] Top shape: 100 64 8 8 (409600)
I0331 11:06:34.710378 24302 net.cpp:165] Memory required for data: 31540400
I0331 11:06:34.710382 24302 layer_factory.hpp:77] Creating layer pool3
I0331 11:06:34.710391 24302 net.cpp:100] Creating Layer pool3
I0331 11:06:34.710395 24302 net.cpp:434] pool3 <- conv3
I0331 11:06:34.710402 24302 net.cpp:408] pool3 -> pool3
I0331 11:06:34.713034 24302 net.cpp:150] Setting up pool3
I0331 11:06:34.713042 24302 net.cpp:157] Top shape: 100 64 4 4 (102400)
I0331 11:06:34.713068 24302 net.cpp:165] Memory required for data: 31950000
I0331 11:06:34.713074 24302 layer_factory.hpp:77] Creating layer ip1
I0331 11:06:34.713083 24302 net.cpp:100] Creating Layer ip1
I0331 11:06:34.713088 24302 net.cpp:434] ip1 <- pool3
I0331 11:06:34.713095 24302 net.cpp:408] ip1 -> ip1
I0331 11:06:34.714015 24302 net.cpp:150] Setting up ip1
I0331 11:06:34.714020 24302 net.cpp:157] Top shape: 100 64 (6400)
I0331 11:06:34.714026 24302 net.cpp:165] Memory required for data: 31975600
I0331 11:06:34.714033 24302 layer_factory.hpp:77] Creating layer ip2
I0331 11:06:34.714041 24302 net.cpp:100] Creating Layer ip2
I0331 11:06:34.714046 24302 net.cpp:434] ip2 <- ip1
I0331 11:06:34.714053 24302 net.cpp:408] ip2 -> ip2
I0331 11:06:34.714442 24302 net.cpp:150] Setting up ip2
I0331 11:06:34.714448 24302 net.cpp:157] Top shape: 100 10 (1000)
I0331 11:06:34.714453 24302 net.cpp:165] Memory required for data: 31979600
I0331 11:06:34.714462 24302 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
I0331 11:06:34.714468 24302 net.cpp:100] Creating Layer ip2_ip2_0_split
I0331 11:06:34.714474 24302 net.cpp:434] ip2_ip2_0_split <- ip2
I0331 11:06:34.714480 24302 net.cpp:408] ip2_ip2_0_split -> ip2_ip2_0_split_0
I0331 11:06:34.714489 24302 net.cpp:408] ip2_ip2_0_split -> ip2_ip2_0_split_1
I0331 11:06:34.714628 24302 net.cpp:150] Setting up ip2_ip2_0_split
I0331 11:06:34.714634 24302 net.cpp:157] Top shape: 100 10 (1000)
I0331 11:06:34.714639 24302 net.cpp:157] Top shape: 100 10 (1000)
I0331 11:06:34.714644 24302 net.cpp:165] Memory required for data: 31987600
I0331 11:06:34.714650 24302 layer_factory.hpp:77] Creating layer accuracy
I0331 11:06:34.714658 24302 net.cpp:100] Creating Layer accuracy
I0331 11:06:34.714664 24302 net.cpp:434] accuracy <- ip2_ip2_0_split_0
I0331 11:06:34.714669 24302 net.cpp:434] accuracy <- label_cifar_1_split_0
I0331 11:06:34.714676 24302 net.cpp:408] accuracy -> accuracy
I0331 11:06:34.714685 24302 net.cpp:150] Setting up accuracy
I0331 11:06:34.714690 24302 net.cpp:157] Top shape: (1)
I0331 11:06:34.714695 24302 net.cpp:165] Memory required for data: 31987604
I0331 11:06:34.714699 24302 layer_factory.hpp:77] Creating layer loss
I0331 11:06:34.714705 24302 net.cpp:100] Creating Layer loss
I0331 11:06:34.714710 24302 net.cpp:434] loss <- ip2_ip2_0_split_1
I0331 11:06:34.714715 24302 net.cpp:434] loss <- label_cifar_1_split_1
I0331 11:06:34.714721 24302 net.cpp:408] loss -> loss
I0331 11:06:34.714730 24302 layer_factory.hpp:77] Creating layer loss
I0331 11:06:34.717491 24302 net.cpp:150] Setting up loss
I0331 11:06:34.717497 24302 net.cpp:157] Top shape: (1)
I0331 11:06:34.717502 24302 net.cpp:160]     with loss weight 1
I0331 11:06:34.717511 24302 net.cpp:165] Memory required for data: 31987608
I0331 11:06:34.717516 24302 net.cpp:226] loss needs backward computation.
I0331 11:06:34.717522 24302 net.cpp:228] accuracy does not need backward computation.
I0331 11:06:34.717527 24302 net.cpp:226] ip2_ip2_0_split needs backward computation.
I0331 11:06:34.717532 24302 net.cpp:226] ip2 needs backward computation.
I0331 11:06:34.717537 24302 net.cpp:226] ip1 needs backward computation.
I0331 11:06:34.717542 24302 net.cpp:226] pool3 needs backward computation.
I0331 11:06:34.717547 24302 net.cpp:226] relu3 needs backward computation.
I0331 11:06:34.717551 24302 net.cpp:226] conv3 needs backward computation.
I0331 11:06:34.717556 24302 net.cpp:226] pool2 needs backward computation.
I0331 11:06:34.717561 24302 net.cpp:226] relu2 needs backward computation.
I0331 11:06:34.717566 24302 net.cpp:226] conv2 needs backward computation.
I0331 11:06:34.717571 24302 net.cpp:226] relu1 needs backward computation.
I0331 11:06:34.717576 24302 net.cpp:226] pool1 needs backward computation.
I0331 11:06:34.717579 24302 net.cpp:226] conv1 needs backward computation.
I0331 11:06:34.717584 24302 net.cpp:228] label_cifar_1_split does not need backward computation.
I0331 11:06:34.717591 24302 net.cpp:228] cifar does not need backward computation.
I0331 11:06:34.717595 24302 net.cpp:270] This network produces output accuracy
I0331 11:06:34.717609 24302 net.cpp:270] This network produces output loss
I0331 11:06:34.717622 24302 net.cpp:283] Network initialization done.
I0331 11:06:34.717659 24302 solver.cpp:60] Solver scaffolding done.
I0331 11:06:34.719099 24302 caffe.cpp:251] Starting Optimization
I0331 11:06:34.719105 24302 solver.cpp:279] Solving CIFAR10_quick
I0331 11:06:34.719110 24302 solver.cpp:280] Learning Rate Policy: fixed
I0331 11:06:34.719523 24302 solver.cpp:337] Iteration 0, Testing net (#0)
I0331 11:06:34.916335 24302 solver.cpp:404]     Test net output #0: accuracy = 0.0963
I0331 11:06:34.916357 24302 solver.cpp:404]     Test net output #1: loss = 2.3027 (* 1 = 2.3027 loss)
I0331 11:06:34.923990 24302 solver.cpp:228] Iteration 0, loss = 230.337
I0331 11:06:34.924002 24302 solver.cpp:244]     Train net output #0: loss = 2.30337 (* 1 = 2.30337 loss)
I0331 11:06:34.924007 24302 sgd_solver.cpp:106] Iteration 0, lr = 0.001
I0331 11:06:35.254720 24302 solver.cpp:228] Iteration 100, loss = 1.69312
I0331 11:06:35.254740 24302 solver.cpp:244]     Train net output #0: loss = 1.69312 (* 1 = 1.69312 loss)
I0331 11:06:35.254745 24302 sgd_solver.cpp:106] Iteration 100, lr = 0.001
I0331 11:06:35.590265 24302 solver.cpp:228] Iteration 200, loss = 1.70744
I0331 11:06:35.590287 24302 solver.cpp:244]     Train net output #0: loss = 1.70743 (* 1 = 1.70743 loss)
I0331 11:06:35.590291 24302 sgd_solver.cpp:106] Iteration 200, lr = 0.001
I0331 11:06:35.923384 24302 solver.cpp:228] Iteration 300, loss = 1.26461
I0331 11:06:35.923406 24302 solver.cpp:244]     Train net output #0: loss = 1.26461 (* 1 = 1.26461 loss)
I0331 11:06:35.923410 24302 sgd_solver.cpp:106] Iteration 300, lr = 0.001
I0331 11:06:36.248895 24302 solver.cpp:228] Iteration 400, loss = 1.36565
I0331 11:06:36.248917 24302 solver.cpp:244]     Train net output #0: loss = 1.36565 (* 1 = 1.36565 loss)
I0331 11:06:36.248922 24302 sgd_solver.cpp:106] Iteration 400, lr = 0.001
I0331 11:06:36.579169 24302 solver.cpp:337] Iteration 500, Testing net (#0)
I0331 11:06:36.751092 24302 solver.cpp:404]     Test net output #0: accuracy = 0.5509
I0331 11:06:36.751114 24302 solver.cpp:404]     Test net output #1: loss = 1.27784 (* 1 = 1.27784 loss)
I0331 11:06:36.754051 24302 solver.cpp:228] Iteration 500, loss = 1.25145
I0331 11:06:36.754062 24302 solver.cpp:244]     Train net output #0: loss = 1.25144 (* 1 = 1.25144 loss)
I0331 11:06:36.754066 24302 sgd_solver.cpp:106] Iteration 500, lr = 0.001
I0331 11:06:37.088784 24302 solver.cpp:228] Iteration 600, loss = 1.25302
I0331 11:06:37.088806 24302 solver.cpp:244]     Train net output #0: loss = 1.25302 (* 1 = 1.25302 loss)
I0331 11:06:37.088811 24302 sgd_solver.cpp:106] Iteration 600, lr = 0.001
I0331 11:06:37.420756 24302 solver.cpp:228] Iteration 700, loss = 1.2073
I0331 11:06:37.420778 24302 solver.cpp:244]     Train net output #0: loss = 1.20729 (* 1 = 1.20729 loss)
I0331 11:06:37.420784 24302 sgd_solver.cpp:106] Iteration 700, lr = 0.001
I0331 11:06:37.749595 24302 solver.cpp:228] Iteration 800, loss = 1.08577
I0331 11:06:37.749617 24302 solver.cpp:244]     Train net output #0: loss = 1.08576 (* 1 = 1.08576 loss)
I0331 11:06:37.749622 24302 sgd_solver.cpp:106] Iteration 800, lr = 0.001
I0331 11:06:38.087330 24302 solver.cpp:228] Iteration 900, loss = 0.979316
I0331 11:06:38.087352 24302 solver.cpp:244]     Train net output #0: loss = 0.979311 (* 1 = 0.979311 loss)
I0331 11:06:38.087357 24302 sgd_solver.cpp:106] Iteration 900, lr = 0.001
I0331 11:06:38.420168 24302 solver.cpp:337] Iteration 1000, Testing net (#0)
I0331 11:06:38.593856 24302 solver.cpp:404]     Test net output #0: accuracy = 0.6054
I0331 11:06:38.593878 24302 solver.cpp:404]     Test net output #1: loss = 1.1508 (* 1 = 1.1508 loss)
I0331 11:06:38.596915 24302 solver.cpp:228] Iteration 1000, loss = 1.03848
I0331 11:06:38.596928 24302 solver.cpp:244]     Train net output #0: loss = 1.03847 (* 1 = 1.03847 loss)
I0331 11:06:38.596935 24302 sgd_solver.cpp:106] Iteration 1000, lr = 0.001
I0331 11:06:38.926901 24302 solver.cpp:228] Iteration 1100, loss = 1.06011
I0331 11:06:38.926940 24302 solver.cpp:244]     Train net output #0: loss = 1.06011 (* 1 = 1.06011 loss)
I0331 11:06:38.926945 24302 sgd_solver.cpp:106] Iteration 1100, lr = 0.001
I0331 11:06:39.255908 24302 solver.cpp:228] Iteration 1200, loss = 0.967729
I0331 11:06:39.255928 24302 solver.cpp:244]     Train net output #0: loss = 0.967723 (* 1 = 0.967723 loss)
I0331 11:06:39.255934 24302 sgd_solver.cpp:106] Iteration 1200, lr = 0.001
I0331 11:06:39.587491 24302 solver.cpp:228] Iteration 1300, loss = 0.873639
I0331 11:06:39.587512 24302 solver.cpp:244]     Train net output #0: loss = 0.873634 (* 1 = 0.873634 loss)
I0331 11:06:39.587517 24302 sgd_solver.cpp:106] Iteration 1300, lr = 0.001
I0331 11:06:39.916858 24302 solver.cpp:228] Iteration 1400, loss = 0.822912
I0331 11:06:39.916877 24302 solver.cpp:244]     Train net output #0: loss = 0.822906 (* 1 = 0.822906 loss)
I0331 11:06:39.916882 24302 sgd_solver.cpp:106] Iteration 1400, lr = 0.001
I0331 11:06:40.243862 24302 solver.cpp:337] Iteration 1500, Testing net (#0)
I0331 11:06:40.418777 24302 solver.cpp:404]     Test net output #0: accuracy = 0.6428
I0331 11:06:40.418798 24302 solver.cpp:404]     Test net output #1: loss = 1.03695 (* 1 = 1.03695 loss)
I0331 11:06:40.422040 24302 solver.cpp:228] Iteration 1500, loss = 0.917664
I0331 11:06:40.422051 24302 solver.cpp:244]     Train net output #0: loss = 0.917658 (* 1 = 0.917658 loss)
I0331 11:06:40.422056 24302 sgd_solver.cpp:106] Iteration 1500, lr = 0.001
I0331 11:06:40.751153 24302 solver.cpp:228] Iteration 1600, loss = 0.951443
I0331 11:06:40.751173 24302 solver.cpp:244]     Train net output #0: loss = 0.951437 (* 1 = 0.951437 loss)
I0331 11:06:40.751178 24302 sgd_solver.cpp:106] Iteration 1600, lr = 0.001
I0331 11:06:41.082576 24302 solver.cpp:228] Iteration 1700, loss = 0.824344
I0331 11:06:41.082597 24302 solver.cpp:244]     Train net output #0: loss = 0.824338 (* 1 = 0.824338 loss)
I0331 11:06:41.082602 24302 sgd_solver.cpp:106] Iteration 1700, lr = 0.001
I0331 11:06:41.412235 24302 solver.cpp:228] Iteration 1800, loss = 0.814171
I0331 11:06:41.412256 24302 solver.cpp:244]     Train net output #0: loss = 0.814166 (* 1 = 0.814166 loss)
I0331 11:06:41.412261 24302 sgd_solver.cpp:106] Iteration 1800, lr = 0.001
I0331 11:06:41.743070 24302 solver.cpp:228] Iteration 1900, loss = 0.746516
I0331 11:06:41.743091 24302 solver.cpp:244]     Train net output #0: loss = 0.74651 (* 1 = 0.74651 loss)
I0331 11:06:41.743096 24302 sgd_solver.cpp:106] Iteration 1900, lr = 0.001
I0331 11:06:42.072156 24302 solver.cpp:337] Iteration 2000, Testing net (#0)
I0331 11:06:42.247408 24302 solver.cpp:404]     Test net output #0: accuracy = 0.6774
I0331 11:06:42.247429 24302 solver.cpp:404]     Test net output #1: loss = 0.943321 (* 1 = 0.943321 loss)
I0331 11:06:42.250512 24302 solver.cpp:228] Iteration 2000, loss = 0.796382
I0331 11:06:42.250524 24302 solver.cpp:244]     Train net output #0: loss = 0.796376 (* 1 = 0.796376 loss)
I0331 11:06:42.250528 24302 sgd_solver.cpp:106] Iteration 2000, lr = 0.001
I0331 11:06:42.582597 24302 solver.cpp:228] Iteration 2100, loss = 0.88666
I0331 11:06:42.582618 24302 solver.cpp:244]     Train net output #0: loss = 0.886655 (* 1 = 0.886655 loss)
I0331 11:06:42.582623 24302 sgd_solver.cpp:106] Iteration 2100, lr = 0.001
I0331 11:06:42.911408 24302 solver.cpp:228] Iteration 2200, loss = 0.76949
I0331 11:06:42.911428 24302 solver.cpp:244]     Train net output #0: loss = 0.769484 (* 1 = 0.769484 loss)
I0331 11:06:42.911433 24302 sgd_solver.cpp:106] Iteration 2200, lr = 0.001
I0331 11:06:43.243353 24302 solver.cpp:228] Iteration 2300, loss = 0.739908
I0331 11:06:43.243374 24302 solver.cpp:244]     Train net output #0: loss = 0.739902 (* 1 = 0.739902 loss)
I0331 11:06:43.243379 24302 sgd_solver.cpp:106] Iteration 2300, lr = 0.001
I0331 11:06:43.573737 24302 solver.cpp:228] Iteration 2400, loss = 0.757853
I0331 11:06:43.573757 24302 solver.cpp:244]     Train net output #0: loss = 0.757848 (* 1 = 0.757848 loss)
I0331 11:06:43.573762 24302 sgd_solver.cpp:106] Iteration 2400, lr = 0.001
I0331 11:06:43.900631 24302 solver.cpp:337] Iteration 2500, Testing net (#0)
I0331 11:06:44.083359 24302 solver.cpp:404]     Test net output #0: accuracy = 0.6871
I0331 11:06:44.083381 24302 solver.cpp:404]     Test net output #1: loss = 0.925407 (* 1 = 0.925407 loss)
I0331 11:06:44.086453 24302 solver.cpp:228] Iteration 2500, loss = 0.748057
I0331 11:06:44.086467 24302 solver.cpp:244]     Train net output #0: loss = 0.748051 (* 1 = 0.748051 loss)
I0331 11:06:44.086472 24302 sgd_solver.cpp:106] Iteration 2500, lr = 0.001
I0331 11:06:44.417397 24302 solver.cpp:228] Iteration 2600, loss = 0.827012
I0331 11:06:44.417420 24302 solver.cpp:244]     Train net output #0: loss = 0.827007 (* 1 = 0.827007 loss)
I0331 11:06:44.417425 24302 sgd_solver.cpp:106] Iteration 2600, lr = 0.001
I0331 11:06:44.752656 24302 solver.cpp:228] Iteration 2700, loss = 0.765873
I0331 11:06:44.752678 24302 solver.cpp:244]     Train net output #0: loss = 0.765867 (* 1 = 0.765867 loss)
I0331 11:06:44.752684 24302 sgd_solver.cpp:106] Iteration 2700, lr = 0.001
I0331 11:06:45.087687 24302 solver.cpp:228] Iteration 2800, loss = 0.689853
I0331 11:06:45.087709 24302 solver.cpp:244]     Train net output #0: loss = 0.689848 (* 1 = 0.689848 loss)
I0331 11:06:45.087714 24302 sgd_solver.cpp:106] Iteration 2800, lr = 0.001
I0331 11:06:45.419508 24302 solver.cpp:228] Iteration 2900, loss = 0.726292
I0331 11:06:45.419528 24302 solver.cpp:244]     Train net output #0: loss = 0.726286 (* 1 = 0.726286 loss)
I0331 11:06:45.419533 24302 sgd_solver.cpp:106] Iteration 2900, lr = 0.001
I0331 11:06:45.754101 24302 solver.cpp:337] Iteration 3000, Testing net (#0)
I0331 11:06:45.933135 24302 solver.cpp:404]     Test net output #0: accuracy = 0.6946
I0331 11:06:45.933156 24302 solver.cpp:404]     Test net output #1: loss = 0.898484 (* 1 = 0.898484 loss)
I0331 11:06:45.936143 24302 solver.cpp:228] Iteration 3000, loss = 0.690615
I0331 11:06:45.936157 24302 solver.cpp:244]     Train net output #0: loss = 0.69061 (* 1 = 0.69061 loss)
I0331 11:06:45.936163 24302 sgd_solver.cpp:106] Iteration 3000, lr = 0.001
I0331 11:06:46.273326 24302 solver.cpp:228] Iteration 3100, loss = 0.794096
I0331 11:06:46.273349 24302 solver.cpp:244]     Train net output #0: loss = 0.79409 (* 1 = 0.79409 loss)
I0331 11:06:46.273353 24302 sgd_solver.cpp:106] Iteration 3100, lr = 0.001
I0331 11:06:46.607097 24302 solver.cpp:228] Iteration 3200, loss = 0.695419
I0331 11:06:46.607121 24302 solver.cpp:244]     Train net output #0: loss = 0.695413 (* 1 = 0.695413 loss)
I0331 11:06:46.607129 24302 sgd_solver.cpp:106] Iteration 3200, lr = 0.001
I0331 11:06:46.940327 24302 solver.cpp:228] Iteration 3300, loss = 0.636181
I0331 11:06:46.940351 24302 solver.cpp:244]     Train net output #0: loss = 0.636175 (* 1 = 0.636175 loss)
I0331 11:06:46.940356 24302 sgd_solver.cpp:106] Iteration 3300, lr = 0.001
I0331 11:06:47.278142 24302 solver.cpp:228] Iteration 3400, loss = 0.686613
I0331 11:06:47.278164 24302 solver.cpp:244]     Train net output #0: loss = 0.686607 (* 1 = 0.686607 loss)
I0331 11:06:47.278169 24302 sgd_solver.cpp:106] Iteration 3400, lr = 0.001
I0331 11:06:47.610103 24302 solver.cpp:337] Iteration 3500, Testing net (#0)
I0331 11:06:47.792064 24302 solver.cpp:404]     Test net output #0: accuracy = 0.6998
I0331 11:06:47.792084 24302 solver.cpp:404]     Test net output #1: loss = 0.88248 (* 1 = 0.88248 loss)
I0331 11:06:47.795045 24302 solver.cpp:228] Iteration 3500, loss = 0.638955
I0331 11:06:47.795059 24302 solver.cpp:244]     Train net output #0: loss = 0.63895 (* 1 = 0.63895 loss)
I0331 11:06:47.795064 24302 sgd_solver.cpp:106] Iteration 3500, lr = 0.001
I0331 11:06:48.126653 24302 solver.cpp:228] Iteration 3600, loss = 0.733167
I0331 11:06:48.126678 24302 solver.cpp:244]     Train net output #0: loss = 0.733161 (* 1 = 0.733161 loss)
I0331 11:06:48.126683 24302 sgd_solver.cpp:106] Iteration 3600, lr = 0.001
I0331 11:06:48.459372 24302 solver.cpp:228] Iteration 3700, loss = 0.649733
I0331 11:06:48.459393 24302 solver.cpp:244]     Train net output #0: loss = 0.649728 (* 1 = 0.649728 loss)
I0331 11:06:48.459398 24302 sgd_solver.cpp:106] Iteration 3700, lr = 0.001
I0331 11:06:48.793612 24302 solver.cpp:228] Iteration 3800, loss = 0.619229
I0331 11:06:48.793632 24302 solver.cpp:244]     Train net output #0: loss = 0.619223 (* 1 = 0.619223 loss)
I0331 11:06:48.793637 24302 sgd_solver.cpp:106] Iteration 3800, lr = 0.001
I0331 11:06:49.126967 24302 solver.cpp:228] Iteration 3900, loss = 0.657576
I0331 11:06:49.126991 24302 solver.cpp:244]     Train net output #0: loss = 0.65757 (* 1 = 0.65757 loss)
I0331 11:06:49.126996 24302 sgd_solver.cpp:106] Iteration 3900, lr = 0.001
I0331 11:06:49.457964 24302 solver.cpp:464] Snapshotting to HDF5 file examples/cifar10/cifar10_quick_iter_4000.caffemodel.h5
I0331 11:06:49.691646 24302 sgd_solver.cpp:283] Snapshotting solver state to HDF5 file examples/cifar10/cifar10_quick_iter_4000.solverstate.h5
I0331 11:06:49.697353 24302 solver.cpp:317] Iteration 4000, loss = 0.569912
I0331 11:06:49.697367 24302 solver.cpp:337] Iteration 4000, Testing net (#0)
I0331 11:06:49.878609 24302 solver.cpp:404]     Test net output #0: accuracy = 0.7054
I0331 11:06:49.878630 24302 solver.cpp:404]     Test net output #1: loss = 0.863621 (* 1 = 0.863621 loss)
I0331 11:06:49.878636 24302 solver.cpp:322] Optimization Done.
I0331 11:06:49.878639 24302 caffe.cpp:254] Optimization Done.

```




### Make sure OpenCL AMD GPU devices

```
/opt/rocm/opencl/bin/x86_64/clinfo
```

```
johndoe@local:~$ /opt/rocm/opencl/bin/x86_64/clinfo

Number of platforms:				 1
  Platform Profile:				 FULL_PROFILE
  Platform Version:				 OpenCL 2.1 AMD-APP (2679.0)
  Platform Name:				 AMD Accelerated Parallel Processing
  Platform Vendor:				 Advanced Micro Devices, Inc.
  Platform Extensions:				 cl_khr_icd cl_amd_event_callback 


  Platform Name:				 AMD Accelerated Parallel Processing
Number of devices:				 1
  Device Type:					 CL_DEVICE_TYPE_GPU
  Vendor ID:					 1002h
  Board name:					 Device 687f
  Device Topology:				 PCI[ B#4, D#0, F#0 ]
  Max compute units:				 64
  Max work items dimensions:			 3
    Max work items[0]:				 1024
    Max work items[1]:				 1024
    Max work items[2]:				 1024
  Max work group size:				 256
  Preferred vector width char:			 4
  Preferred vector width short:			 2
  Preferred vector width int:			 1
  Preferred vector width long:			 1
  Preferred vector width float:			 1
  Preferred vector width double:		 1
  Native vector width char:			 4
  Native vector width short:			 2
  Native vector width int:			 1
  Native vector width long:			 1
  Native vector width float:			 1
  Native vector width double:			 1
  Max clock frequency:				 1630Mhz
  Address bits:					 64
  Max memory allocation:			 7287183769
  Image support:				 Yes
  Max number of images read arguments:		 128
  Max number of images write arguments:		 8
  Max image 2D width:				 16384
  Max image 2D height:				 16384
  Max image 3D width:				 2048
  Max image 3D height:				 2048
  Max image 3D depth:				 2048
  Max samplers within kernel:			 26751
  Max size of kernel argument:			 1024
  Alignment (bits) of base address:		 1024
  Minimum alignment (bytes) for any datatype:	 128
  Single precision floating point capability
    Denorms:					 Yes
    Quiet NaNs:					 Yes
    Round to nearest even:			 Yes
    Round to zero:				 Yes
    Round to +ve and infinity:			 Yes
    IEEE754-2008 fused multiply-add:		 Yes
  Cache type:					 Read/Write
  Cache line size:				 64
  Cache size:					 16384
  Global memory size:				 8573157376
  Constant buffer size:				 7287183769
  Max number of constant args:			 8
  Local memory type:				 Scratchpad
  Local memory size:				 65536
  Max pipe arguments:				 16
  Max pipe active reservations:			 16
  Max pipe packet size:				 2992216473
  Max global variable size:			 7287183769
  Max global variable preferred total size:	 8573157376
  Max read/write image args:			 64
  Max on device events:				 1024
  Queue on device max size:			 8388608
  Max on device queues:				 1
  Queue on device preferred size:		 262144
  SVM capabilities:				 
    Coarse grain buffer:			 Yes
    Fine grain buffer:				 Yes
    Fine grain system:				 No
    Atomics:					 No
  Preferred platform atomic alignment:		 0
  Preferred global atomic alignment:		 0
  Preferred local atomic alignment:		 0
  Kernel Preferred work group size multiple:	 64
  Error correction support:			 0
  Unified memory for Host and Device:		 0
  Profiling timer resolution:			 1
  Device endianess:				 Little
  Available:					 Yes
  Compiler available:				 Yes
  Execution capabilities:				 
    Execute OpenCL kernels:			 Yes
    Execute native function:			 No
  Queue on Host properties:				 
    Out-of-Order:				 No
    Profiling :					 Yes
  Queue on Device properties:				 
    Out-of-Order:				 Yes
    Profiling :					 Yes
  Platform ID:					 0x7efc84a47df0
  Name:						 gfx900
  Vendor:					 Advanced Micro Devices, Inc.
  Device OpenCL C version:			 OpenCL C 2.0 
  Driver version:				 2679.0 (HSA1.1,LC)
  Profile:					 FULL_PROFILE
  Version:					 OpenCL 1.2 
  Extensions:					 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_gl_sharing cl_amd_device_attribute_query cl_amd_media_ops cl_amd_media_ops2 cl_khr_subgroups cl_khr_depth_images cl_amd_copy_buffer_p2p cl_amd_assembly_program 
  
```




<br>
<br>


### Also see, AMDGPU - ROCm Caffe/PyTorch/Tensorflow 1.x installation, official, introduction on docker
- GPUEater ROCM-Tensorflow installation https://www.gpueater.com/help
- GPUEater github ROCm-Tensorflow information https://github.com/aieater/rocm_tensorflow_info
- GPUEater github ROCm-PyTorch information https://github.com/aieater/rocm_pytorch_informations
- GPUEater github ROCm-Caffe information https://github.com/aieater/rocm_caffe_informations
- ROCm+DeepLearning libraries https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html
- ROCm github https://github.com/RadeonOpenCompute/ROCm
- ROCm-TensorFlow on Docker https://hub.docker.com/r/rocm/tensorflow/
