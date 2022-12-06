This is a simple demo of Human Pose Driven Object Effects Recommendation. The paper is available at https://arxiv.org/abs/2209.08353

The input is a pose array of a micro-video and many pre-defined items, the output is the top 5 recommendation list.

The pre-defined data include:
- '70_sample.pkl': the pose tensor, (N, T, V, C), N: the number of the samples, T: the length of the time window, V: the number of the pose landmark (default is 33 in Blazepose), C: the number of channels (default is 4: x, y, z, and visibility) --> (5, 10, 33, 4)
- 'item_data': item name of pre-defined items --> 221
- 'item_feature.pkl': item representation of pre-defined items (num_cat, num_item, emb_size_prototype) --> (4, 221, 64)
- 'prototype_dis.npy': prototype distribution on prototypes (num_item, num_cat) --> (221, 4)

The trained model:
- 'best_model_pose.pt': the trained pose model
  - input: pose tensor (N, C, T, V, M), M: the number of person, default is 1. --> (5, 4, 10, 33, 1)
  - output: video (pose) representation, (num_cat, num_pose_sample, emb_size_prototype) --> (4, 5, 64)

For detailed model implementation, please refer to './model'
