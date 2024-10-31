 
# gpu 사용량 실시간 조회

* 1초에 한번씩 업데이트 해줌
```
nvidia-smi -l 1
```

* 0.5초 간격으로 Diff에 대해서 Highlight 해서 보여줌
```
watch -d -n 0.5 nvidia-smi 
```



# Docker 관련

```
docker container run -it -d -p 33336:8888 --gpus all --shm-size=128G -v /mnt/sda/jeahun:/code --name demoire 46d26ec802e4 /bin/bash
```

docker container run -it -d --gpus all --shm-size=128G -v /mnt/sda/jeahun:/code --name vd c7e20104018e /bin/bash
# Anaconda 설치
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
```


# Ubuntu 기본 명령어 관련
## scp 명령어 관련
* dropbox, baidu 같은 경우 wget, curl 등으로 다운로드가 안되기에 서버에서 바로 다운이 불가.
* 따라서 우선 mac 에 다운로드 한 후 scp 명령어를 통해 서버로 전송하는 방법 채택.
* 또한 특정 서버에서 다른 서버로 데이터를 전송할 때도 사용가능
* [ref link](https://velog.io/@3436rngus/%EC%84%9C%EB%B2%84-%ED%8C%8C%EC%9D%BC-%EB%A1%9C%EC%BB%AC%EB%A1%9C-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0)
```
# command
scp -P [port 번호] [전송하고자 하는 파일 경로] [user]@[ip]:[다운받고자 하는 서버의 폴더 위치]

# example
scp -P 7722 Data_v1.zip jeahun@165.194.68.7:/mnt/sda/jeahun/datasets

# 폴더일 경우
scp -r -P [port 번호] [전송하고자 하는 파일 경로(mac 의 경로)] [user]@[ip]:[다운받고자 하는 서버의 폴더 위치]
```
## rar 파일 압축해제

```
# 압축한 디렉터리 구조 그대로 현재 경로로 압축 해제 
 unrar x file.rar 
  
 # 압축한 디렉터리 구조 그대로 다른 경로에 압축 해제 
 unrar x file.rar /home/temp 
  
 # 현재 경로에 압축 해제 
 unrar e file.rar 
  
 # 다른 경로에 압축 해제 
 unrar e file.rar /home/temp
출처: https://betwe.tistory.com/entry/리눅스-unrar-설치-및-사용법 [개발과 육아사이:티스토리]
```

## zip 파일 압축해제

```
# 현재 경로 안에 파일이 있는 경우
unzip [파일명].zip

# 특정 경로에 압축을 풀고싶은 경우
unzip [파일명].zip -d [폴더 경로]
```

## mv 명령어 관련

```
mv [옵션] [이동시킬 디렉토리/파일] [이동 될 위치]

ex) mv log.txt folder
// 현재 디렉토리의 log.txt 파일을 folder 디렉토리로 이동

ex) mv log.txt log2.txt
// 현재 디렉토리의 log.txt 파일의 이름을 log2.txt로 변경

ex) mv /app/bin/logs/log.txt /app/dw
// /app/bin/logs 디렉토리의 log.txt 파일을 /app/dw 디렉토리로 이동
```

* [reference link](https://code-lab1.tistory.com/306) 

cp -r /code/datasets/TMM22/testset /code/RRID/dataset/testset
## cp 명령어 관련
```
# cp 명령어 기본 사용 방법
cp [옵션] [복사 대상 디렉터리or파일] [복사될 디렉터리or파일]

# ex1) 현재 디렉터리에 있는 test.js파일을 현재 디렉터리에 test_backup.js이라는 이름으로 변경 후 복사
cp test.js test_backup.js

# ex2) /hw/js/ 경로에 있는 test.js 파일을 /backup/js/ 경로에 복사
cp /hw/js/test.js /backup/js/test.js

# ex3) 폴더를 복사할 경우 r 옵션 추가
cp -r /hw/js /backup/js
```

* [reference link](https://backendcode.tistory.com/307)

# TMUX 사용법

```bash
# 새로운 세션 생성
tmux new -s (session_name)

# 세션 만들면서 윈도우랑 같이 생성
tmux new -s (session_name) -n (window_name)

# 세션 종료
exit

# 세션 목록
tmux ls

# 세션 다시 시작하기(다시 불러오기)
tmux attach -t session_number

# 세션 중단하기
(ctrl + b) d

# 스크롤하기
ctrl + b + [

# 특정 세션 강제 종료
tmux kill-session -t session_number
```

* [reference link](https://velog.io/@ur-luella/tmux-%EC%82%AC%EC%9A%A9%EB%B2%95)
