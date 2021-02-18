from bayes_act_max.datasets.get_datasets import *

batch_size = 50
trainloader, valloader, testloader, testset, idx_to_class, class_to_name = \
  load_tiny_imagenet(batch_size=batch_size)