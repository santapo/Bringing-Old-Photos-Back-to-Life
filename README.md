


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

#### RUN
```
### pwd: Bringing-old-photos-back-to-life
cd Global
mlchain run # check swagger at port 8001
```