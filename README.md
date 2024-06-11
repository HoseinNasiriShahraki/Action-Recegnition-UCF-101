### Action Recognition With Pytorch and CNN 


##### UCF-101 Dataset(https://www.crcv.ucf.edu/research/data-sets/ucf101/)
![firstpic](https://github.com/HoseinNasiriShahraki/Action-Recognition-UCF-101/blob/main/159976180-393c0267-6a7f-4098-9f17-6f4722acb3cb.jpeg?raw=true "First-Pic")




```python
UCF_Model(
  (relu): LeakyReLU(negative_slope=0.01)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (dropout2): Dropout(p=0.2, inplace=False)
  (dropout3): Dropout(p=0.3, inplace=False)
  (maxpool): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (batchnorm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (batchnorm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (l1): Linear(in_features=16384, out_features=4096, bias=True)
  (l2): Linear(in_features=4096, out_features=1024, bias=True)
  (batchnorm4): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (l5): Linear(in_features=1024, out_features=101, bias=True)
)

```

Training Results:
```
epoch 30/30, loss = 0.0025690012, accuracy = 98.42

```
