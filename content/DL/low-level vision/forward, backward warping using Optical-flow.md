#low_level_vision

# introduction
---
* video demoireing 관련 실험을 돌리다가 optical-flow 를 이용하여 k 개의 frame 에 대해 align 을 맞춰야 할 필요성을 느꼈다.
* 다만, warping 이 여전히 헷갈리기에 이 기회를 통해 정리를 하고 넘어가고자 한다.
* [reference link](https://www.youtube.com/watch?v=9iN-dAKqcwM&t=6s)


# Forward warping

![[Pasted image 20240911212731.png|500]]

* Forward warping 이란?
	* 위 그림처럼 Original image 의 특정 point $P$ 를 trasnformation matrix $T$ 를 적용하여 $p'$ 으로 warping 하는 것 
$$
T: \mathbb{R}^2\rightarrow\mathbb{R}^2
$$
$$
T(p)=p'
$$
* limitation: 
	* 위의 그림처럼 transformed coordinate $p'$ 은 Transformed image 의 pixel coordinate 의 위치와 일치하지 않는다.

![[Pasted image 20240911213954.png|500]]
* 여기서 $p'$ 에 먼 점일수록 weight 값을 적게 부여 (e.g. 왼쪽 아래가 제일 머니까 제일 작은 가중치를 가짐)
* 즉, $p'$ 의 값은 아래와 같이 정해짐
$$
value(p')=weight*pixel\_value
$$
