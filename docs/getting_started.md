# Train

```shell
bash tools/dist_train.sh configs/OccFusion2.py 8
```

# **Test**

```shell
bash tools/dist_test.sh configs/OccFusion2.py ${path to corresponding checkpoint}$ 8
```

# Training visualization

### tensorboard

tensorboard --logdir ./work_dirs

### swanlab

```
swanlab: Using SwanLab to track your experiments. Please refer to https://docs.swanlab.cn for more information.
swanlab: (1) Create a SwanLab account.
swanlab: (2) Use an existing SwanLab account.
swanlab: (3) Don't visualize my results.
swanlab: Enter your choice:
```
