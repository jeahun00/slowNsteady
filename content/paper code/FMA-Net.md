
# Requirements

* FMA Net github 에 있는 문서 그대로 할 경우 requirement 를 맞추기가 힘들 수 있다. 
* 따라서 내가 겪었던 시행착오들을 기술할 생각이다.
## 1. docker 환경 구축
* docker 는 pytorch 와 cuda 를 동시에 만족하는 버전을 찾을 수 없었기에 아래와 같은 nvidia/cuda 를 사용하였다.
![[Pasted image 20240522205829.png|500]]
* 위의 docker image 를 pull
* 이후 docker run 으로 container 실행
```bash
$ docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
$ docker container run -it -d -p 1234:8888 --gpus all --shm-size=128G -v /mnt/sda/jh:/code --name fmanet 46d6ea3f8fea /bin/bash
```

## 2. torch 환경 구축
* 이 논문에서는 cuda:11.8 을 사용하였다.
* 또한 torch >= 1.9.1 를 사용하였기에 torch 1.12.1 을 사용하였다.
```bash
# docker 진입
$ docker exec -it fmanet /bin/bash

# docker 내부로 진입했으면 pytorch 설치
$ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

# Data Preprocessing

## 1. Data Preprocessing

* FMA Net 에서는 REDS Dataset 을 사용한다.
* 따라서 이 dataset 을 다운받는 방법을 설명하고자 한다.
### 1.1. download REDS dataset
(이 부분은 [이 링크](https://honbul.tistory.com/32)를 참고하였다.)
* [REDS dataset info link](https://seungjunnah.github.io/Datasets/reds.html)
* 위 링크에 접속하여 데이터셋을 하나하나 다 다운받아도 된다.
* 하지만 그 과정이 너무 길고 귀찮기에 위 링크에서 만들어둔 [download_REDS.py](https://gist.github.com/SeungjunNah/b10d369b92840cb8dd2118dd4f41d643) 파일을 이용하고자 한다.
* 과정은 아래와 같다.

1. 위의 파일을 python 파일을 하나 생성하여 코드를 붙여넣는다.
2. 이후 아래 명령어로 python 코드를 실행한다.
```bash
$ python3 download_REDS.py --server snu --all
```
*  위의 코드를 처음 실행하면 에러가 발생한다. 이를 무시하고 다시 똑같은 명령어로 실행하면 정상적으로 다운이된다.
	* 이는 아마 google drive 에 접속하는 과정에서 인증쪽에서 문제가 생기는 듯 한데 정확한 이유는 잘 모르겠다.
* 하지만 위의 `--all` option 에서 <span style='color:var(--mk-color-red)'>하나하나의 데이터셋을 받을 때마다 오류가 생기기에</span> 그냥 하나하나 받는게 더 나을 듯 하다.
```bash
$ python3 download_REDS.py --server snu --train_sharp
$ python3 download_REDS.py --server snu --train_blur
$ python3 download_REDS.py --server snu --train_blur_comp
$ python3 download_REDS.py --server snu --train_sharp_bicubic
$ python3 download_REDS.py --server snu --train_blur_bicubic
$ python3 download_REDS.py --server snu --val_sharp
$ python3 download_REDS.py --server snu --val_blur
$ python3 download_REDS.py --server snu --val_blur_comp
$ python3 download_REDS.py --server snu --val_sharp_bicubic
$ python3 download_REDS.py --server snu --val_blur_bicubic
$ python3 download_REDS.py --server snu --test_blur
$ python3 download_REDS.py --server snu --test_blur_comp
$ python3 download_REDS.py --server snu --test_sharp_bicubic
$ python3 download_REDS.py --server snu --test_blur_bicubic
```

* REDS dataset 이 정상적으로 모두 받아졌다면 아래와 같은 zip 파일들이 생기게 된다.
* 아래 파일들을 모두 압축을 해제 해 준다.
```bash
train_blur.zip
val_blur.zip
test_blur.zip
train_blur_bicubic.zip   
val_blur_bicubic.zip
test_blur_bicubic.zip   
train_blur_comp.zip      
val_blur_comp.zip
test_blur_comp.zip      
train_sharp.zip          
val_sharp.zip
test_sharp_bicubic.zip
train_sharp_bicubic.zip
val_sharp_bicubic.zip
```
* 압축해제
```bash
# zip 파일이 존재하는 폴더로 접근
$ unzip '*.zip'
```
* 압축을 해제한 이후 생성되는 폴더의 구조는 아래와 같다.
```bash
train - train_blur
	  - train_sharp
	  - ...
val - val_blur
	- ...
test - train
	 - ...
```
* 이 논문에서 다루는 구조는 조금 상이하다.
* 따라서 각 `train`, `test`, `val` 폴더에 포함되어 있는 하위 폴더들을 모두 REDS 로 옮겨준다.
```bash
# 폴더 구조
# 현재 폴더: FMA-Net/dataset/REDS/
.
|-- test.text
|-- test_blur
|-- test_blur_bicubic
|-- test_blur_comp
|-- test_sharp_bicubic
|-- train_blur
|-- train_blur_bicubic
|-- train_blur_comp
|-- train_sharp
|-- train_sharp_bicubic
|-- val_blur
|-- val_blur_bicubic
|-- val_blur_comp
|-- val_sharp
`-- val_sharp_bicubic
```

### 1.2. preprocessing REDS to REDS4
* REDS dataset 을 이 논문에서 사용하기 위한 dataset 으로 정제하기 위해 아래 코드를 실행한다.
```bash
# 실행경로: ./FMA-Net/
$ python ./preprocessing/generate_reds4.py
```

* 위의 코드가 정상적으로 실행되면 아래와 같은 폴더가 만들어진다.(REDS4)
![[Pasted image 20240522204136.png|200]]
* 이제 REDS4 가 준비되었고 이 데이터셋에 상응하는 optical flow 를 추출해야 한다.
```bash
$ python ./preprocessing/generate_flow.py --model ./preprocessing/pretrained/raft-sintel.pth --mixed_precision
```
* 뒤에 붙어 있는 `--model` option 은 optical flow 를 생성할 때 어떤 pretrained model 을 사용할지에 대한 명시를 해 주는 부분이다.
* 위의 과정이 끝나면 아래처럼 `train_flow_bicubic` 과 `val_flow_bicubic` 폴더가 생성됨을 알 수 있다.
![[Pasted image 20240522205025.png]]
#### 주의!
* 아래는 이 git 에서 기본으로 제공하는 [generate_flow.py](https://github.com/KAIST-VICLab/FMA-Net/blob/main/preprocessing/generate_flow.py) 의 코드의 일부이다.
```python
### function code ################################
# ...
##################################################

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    generate_flow('./dataset/REDS4/train_sharp_bicubic/train/train_sharp_bicubic/X4')
    generate_flow('./dataset/REDS4/val_sharp_bicubic/val/val_sharp_bicubic/X4')
```
* 위의 코드를 그대로 실행했을 때 경로의 문제가 발생하였다.
* 따라서 이 코드를 아래처럼 수정했을 때 정상적으로 동작했다.
```python
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    generate_flow('./dataset/REDS4/train_sharp_bicubic/X4')
    generate_flow('./dataset/REDS4/val_sharp_bicubic/X4')
```

# Training

```bash
# download code
git clone https://github.com/KAIST-VICLab/FMA-Net
cd FMA-Net

# train FMA-Net on REDS dataset
python main.py --train --config_path experiment.cfg
```
* 이 git 에 있는 그대로 코드를 돌렸을 때 경로 문제가 발생했다.
* 정확히 내가 어떤 실수를 한건지 명확하지 않지만 [data.py](https://github.com/KAIST-VICLab/FMA-Net/blob/main/data.py)를 수정해야 했다.
```python
# 원본 data.py 의 141 번째 줄
    def get_seq_path(self, bath_path):
        seq_list = []
		#######  original part  ###########################################
        dir_list = glob.glob(os.path.join(bath_path, '*/*/*/*'))
        ###################################################################
        for dir in dir_list:
            frame_list = sorted(glob.glob(os.path.join(dir, '*.png')))
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def __len__(self):
        return self.num_data
```
* 위의 코드를 아래처럼 고치면 정상동작했다.
```python
    def get_seq_path(self, bath_path):
        seq_list = []
        #######  modified part  ###########################################
        dir_list = glob.glob(os.path.join(bath_path, '*/*/'))
        ###################################################################
        for dir in dir_list:
            frame_list = sorted(glob.glob(os.path.join(dir, '*.png')))
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def __len__(self):
        return self.num_data
```

* 이제 다시 training code 를 돌렸을 때 아래 같은 화면이 뜨면 성공한 것이다.
![[Pasted image 20240522211219.png|500]]