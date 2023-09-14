# KETI_ar

## Requirements
``` sh
pip install -r requirements.txt
```

## Data preprocessing
Root set up 
``` sh
preprocess_main_new.ipynb
First Block
* data_root 16 ```root = '/share_home/toolkit/datasets/ketivg/'.``` 
```
``` sh
Second Block
* variable 'items' put the object you want```
* Save pickle file ```pickle_file_dir = 'VG230809_pkl/VG_dataset_%s.pkl' %item```
```

## Training & Evaluation
To train the model
Specify root 
``` sh
train_new.ipynb
Third Block
* Cropped image:  ```imagepath = /share_home/toolkit/datasets/ketivg/cropped_images/'```
* Pickel path :  ```picklepath = 'VG230809_pkl/VG_dataset_%s.pkl' %item'```
* Save model of best mA: ```torch.save(model.state_dict(), 'VG230809_pth/%s/best_mA_0829.pth' %item)``` 
```
To evaluate the performance on a validation set

``` sh
visualization_new.ipynb
Second Block
* Put the particular object class : ``` item= 'person' ```
* Cropped image:  ``` '/share_home/toolkit/datasets/ketivg/cropped_images/'' ```
* Pickle path: ``` picklepath = 'VG230809_pkl/VG_dataset_%s.pkl' %item ```
* Model path : ``` state_dict_parallel = torch.load('VG230809_pth/%s/best_mA_0826.pth' %item, map_location=lambda storage, loc: storage) ```
```
