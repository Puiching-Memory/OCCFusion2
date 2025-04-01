# Training and Testing

## NuScenes 3D Semantic Occupancy Prediction Task

**a. Train OCCFusion with 8 GPUs.**

```shell
bash tools/dist_train.sh configs/OccFusion2.py 8
```

**b. Test OCCFusion with 8 GPUs.**

```shell
bash tools/dist_test.sh configs/OccFusion2.py ${path to corresponding checkpoint}$ 8
```


### Training visualization

tensorboard --logdir ./work_dirs
