### 1. Prepare the dataset

The data should be arranged as below tree directory:

```
data
├── phase1
│   ├── annotations
│   ├── PA100k
│   └── PETA
```

### 2. Prepare environment

```
pip install requirement.txt

```

### 3. Training model

Run the below command for training:

```
CUDA_VISIBLE_DEVICES=1 nohup python train_pa100k.py > ./train.log 2>&1 &
```

### 4. Key Parameter Locations in the Model
in ./models/boq.py, you can change L and Q in
```
 def __init__(
        self,
        in_channels=1024,
        proj_channels=1024,
        num_queries=16,
        num_layers=8,
        row_dim=32,
    ):
```

### 5. historical training data logs
you can find it in ./logs_save for pa100k and peta dataset.



