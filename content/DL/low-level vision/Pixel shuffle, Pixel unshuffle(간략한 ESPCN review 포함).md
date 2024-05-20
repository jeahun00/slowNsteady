* REF: 
	* [pixel shuffle 을 처음 제시한 논문](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
		* (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network)
		* [논문리뷰 링크](https://mole-starseeker.tistory.com/84)
	* [pixel shuffle youtube 영상(시각화가 잘 되어 있음)](https://www.youtube.com/watch?v=FqRYbnKXjhE)
	* [transposed conv 에 대해 이전의 conv 부터 설명 잘 되어 있음](https://velog.io/@hayaseleu/Transposed-Convolutional-Layer%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80)
![[qsw4wbxxueub1.gif|500]]
## Conventional upsampling technique
### 1. interpolation
* bilinear, bicubic 등등이 존재
* <span style='color:var(--mk-color-red)'>단점</span>: 원본 image 가 blurry 할 경우 interpolation 이후에도 blurry 하다.
![[Pasted image 20240519150054.png|300]]
### 2. Transposed Convolution
* 아래 그림과 같은 과정을 거친다.
	1. $z, p', s'$ 을 계산한다.
		* $z=s-1$ : zero padding 을 얼마나 줄지에 대한 수치
		* $p'=k-p-1$ : transposed conv 의 padding
		* $s'=1$
	2. input 의 각 row, column 사이에 $z$ 만큼의 0 을 삽입
	3. 2 에서 변형된 input 에 $p'$ 만큼의 0 을 padding 해 준다.
	4. 3 에서 변형된 input 에 stride 가 1인 convolution 연산을 진행한다.
		* 위에서 $s'$ 이 다른 수치라고 하면 stride 를 그 수치만큼 반영하면 됨
![[Pasted image 20240519150526.png]]
* <span style='color:var(--mk-color-red)'>단점</span>: 아래처럼 3x3 형태의 transposed conv 를 적용하면 그 커널 크기만큼의 grid artifact 가 발생
![[Pasted image 20240519150116.png|300]]

### 3. Pixel Shuffle
#### Motivation
* 낮은 해상도의 이미지: LR / 높은 해상도의 이미지: HR
* 기존의 CNN-based Super Resolution 은 아래와 같은 과정을 거친다.(SRCNN, VDSR 등)
	1. bicubic interpolation 을 통해 HR 을 LR 으로 downscaling.
		* 여기서 HR 은 GT: 이걸 $HR_{GT}$ 라고 하자
	2. 이렇게 만들어진 **LR image** 를 다시 **bicubic interpolation** 을 통해 **HR image 로 upscaling** 
		* 이 과정에서 만들어진 HR image 는 train image: 이걸 $HR_{train}$ 이라 하자
	3. 위에서 생성한 image pair-$\{HR_{GT}, HR_{train}\}$ 을 통해 network 를 학습
* 정리하면, image upscaling 이후 image enhancement 진행.

* <mark style='background:var(--mk-color-red)'>위의 과정을 거치는 모델</mark>들은 <span style='color:var(--mk-color-red)'>아래의 단점</span>을 가진다.
	* input image pair 가 처음부터 high resolution 을 가지기에 computational cost 가 높음
	* upscaling 과정에서 bicubic interpolation 을 사용하는데 이는 GT 로의 reconstruction 을 위한 정보를 소실하기에 큰 도움이 되지 않는다.
* 이에 [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf) 논문에서는 low resolution feature 를 통해 점진적으로 high resolution 
* image 를 예측하는 모델을 제시한다.
	* 즉, image enhancement(feature extraction) 이후 image upscaling 진행

#### Method
![[Pasted image 20240520114702.png]]
* 원본 GT High Resolution image 를 bicubic interpolation 을 통해 Low Resolution image 로 변환
* 이렇게 만들어진 LR image 를 2개의 CNN network 를 통과시킨다.
* 이후 upscaling 과정을 거쳐 원본이미지를 예측한다.
* 각 레이어 정보는 아래 참고
	* **1단계 : 특징 추출(feature extraction)**    
	    - 원본 이미지를 그대로 입력 데이터로 사용
	    - num_channel = 1
	    - 5x5 필터
	    - output_channel = 64
	    - Tanh()
	- **2단계 : 특징 추출(feature extraction)**
	
	    - input_channel = 64
	    - 3x3 필터
	    - output_channel = 32
	    - Tanh()
	- **3단계 : 업스케일링(upscaling)**
	    - input_channel = 32
	    - 3x3 필터
	    - output_channel = num_channels * (scale_factor ** 2)
	    - nn.PixelShuffle(scale_factor)  
	        [https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)
	
* 위의 과정에서 3번째 Layer 의 upscaling 시에 사용하는 것이 Pixel Shuffle 이다.
$$
I^{SR}=f^L(I^{LR})
$$
$$\mathbf{I}^{SR} = f^L(\mathbf{I}^{LR}) = \mathcal{PS} \left( \mathbf{W}_L * f^{L-1}(\mathbf{I}^{LR}) + \mathbf{b}_L \right)$$

- $\mathbf{I}^{SR}$ : the super-resolved image.
- $f^{L-1}(\cdot)$ : previous feature map
- $\mathcal{PS}$ : periodic shuffling operation
$$
\mathcal{PS}(T)_{x,y,c} = T_{\left\lfloor \frac{x}{r} \right\rfloor, \left\lfloor \frac{y}{r} \right\rfloor, c \cdot r \cdot \text{mod}(y, r) + c \cdot \text{mod}(x, r)}
$$
- $\mathbf{W}_L$ and $\mathbf{b}_L$ : weights and biases


#### 장점 
* pixel shuffle 에는 learnable 한 network 가 존재. 따라서 데이터에 특화된 upscaling 가능
* 다른 interpolation 방법에 비해 연산량이 적다.
* 다른 upscaling 방법보다 세부적인 detail 을 더 잘 유지한다(reconstruction 정보를 포함할 수 있기 때문)