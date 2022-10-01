


#### Install Dependencies

- Install `python=3.8`
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install mlchain
```

- Download weights
```
bash download-weights
```

#### How to run Inference
```
### pwd: Bringing-old-photos-back-to-life
cd Global
mlchain run # check swagger at port 8001
```

#### How to train

- Check and run [Global/data/Create_Bigfile.py](Global/data/Create_Bigfile.py) to create bigfile of your dataset
```
cd Global/data
python Create_Bigfile.py
```
- Check [Global/data/online_dataset_for_old_photos.py](Global/data/online_dataset_for_old_photos.py) and edit bigfiles name with your own bigfiles name
- Start training by below command
```
cd Global
bash train_domain_A.sh
```
