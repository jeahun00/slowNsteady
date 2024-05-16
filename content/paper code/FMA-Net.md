
# Requirements

* FMA Net github 에 있는 문서 그대로 할 경우 requirement 를 맞추기가 힘들 수 있다. 
* 따라서 내가 겪었던 시행착오들을 기술할 생각이다.
(이 부분은 아직 완벽히 돌려본 게 아니라서 data preprocessing 이 끝나면 다시 업데이트 하도록 하겠다.)
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
python3 download_REDS.py --server snu --all
```
*  위의 코드를 처음 실행하면 에러가 발생한다. 이를 무시하고 다시 똑같은 명령어로 실행하면 정상적으로 다운이된다.
	* 이는 아마 google drive 에 접속하는 과정에서 인증쪽에서 문제가 생기는 듯 한데 정확한 이유는 잘 모르겠다.
* 하지만 위의 `--all` option 에서 하나하나의 데이터셋을 받을 때마다 오류가 생기기에 그냥 하나하나 받는게 더 나을 듯 하다.
```bash
python3 download_REDS.py --server snu --train_sharp
python3 download_REDS.py --server snu --train_blur
python3 download_REDS.py --server snu --train_blur_comp
python3 download_REDS.py --server snu --train_sharp_bicubic
python3 download_REDS.py --server snu --train_blur_bicubic
python3 download_REDS.py --server snu --val_sharp
python3 download_REDS.py --server snu --val_blur
python3 download_REDS.py --server snu --val_blur_comp
python3 download_REDS.py --server snu --val_sharp_bicubic
python3 download_REDS.py --server snu --val_blur_bicubic
python3 download_REDS.py --server snu --test_blur
python3 download_REDS.py --server snu --test_blur_comp
python3 download_REDS.py --server snu --test_sharp_bicubic
python3 download_REDS.py --server snu --test_blur_bicubic
```

