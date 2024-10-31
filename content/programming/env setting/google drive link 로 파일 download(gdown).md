#Programming #Python

* 최근 deep learning 논문의 코드를 계속 돌려보고 있는데 weight(checkpoint or pickle) file 이나 dataset 이 google drive 에 올라가 있는 경우가 많았다.
* 자체에서 다운로드 링크를 제공해주면 wget 이나 curl 로 받으면 되는데 google drive 는 해당 기능을 쓸 수가 없다.
* 따라서 해당 기능을 사용할 수 있는 gdown 사용법에 대해 설명하고자 한다.
* 이 글은 [이 블로그 글](https://code-angie.tistory.com/56)을 참고하여 작성하였다.

## 1. gdown
![[Pasted image 20240621104732.png]]

```bash
pip3 install gdown
# or
pip install gdown
```

1. 구글 드라이브의 open.zip 파일 링크를 공유 받았다면 file_id를 추출한다.
    - https://drive.google.com/file/d/ **16YZxhGfwnvlSLDsfDcaM_Z7nTouqzRaW** /view
    - 주어진 링크에서 file_id는 bold 처리된 부분인 **16YZxhGfwnvlSLDsfDcaM_Z7nTouqzRaW** 이다.
    
* downloads using command line
	* file_id를 "https://drive.google.com/uc?id=" 주소 뒤에 추가한다. 
	* https://drive.google.com/uc?id=[file_id]

```
!gdown https://drive.google.com/uc?id=16YZxhGfwnvlSLDsfDcaM_Z7nTouqzRaW
```

