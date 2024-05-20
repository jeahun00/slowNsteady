
# Abstract

* CNN: large-scale dataset 으로부터 generalize 가 가능한 image prior 를 학습하는데 뛰어남
	* 위와 같은 특성으로 image restoration 과 관련된 작업에서 많이 사용됨
* Transformer: CNN의 한계를 완화하지만 spatial resolution 에 제곱으로 비례하기에 high resolution 을 다루어야 하는 image restorlation 작업에 적용하기 어려움

* 이에 이 논문에서는 multi-head attention and feed-forward network 등을 도입하여 <mark style='background:var(--mk-color-teal)'>long-range pixel 의 상호작용을 가능</mark>케 하며 <mark style='background:var(--mk-color-teal)'>large image 에 대해서도 적용할수 있도록</mark> 하는 모델 제시

# Introduction
* Image restoration 작업은 image 에서 여러 종류의 degradation 을 복원하는 작업
* 여기서 degrade 된 image 의 원본에 해당하는 prior image 의 분포를 찾아야 한다.
* <mark style='background:var(--mk-color-orange)'>Conventional restoration approach</mark>: 
	* <span style='color:var(--mk-color-teal)'>장점</span>: low computational cost
	* <span style='color:var(--mk-color-red)'>단점</span>: low performance(compare to CNN), low generalization
* 위의 방법론을 대체하기 위해 <mark style='background:var(--mk-color-orange)'>CNN-based restoration</mark> 제시
	* <span style='color:var(--mk-color-teal)'>장점</span>: high generalization
	* <span style='color:var(--mk-color-red)'>단점</span>: long-range pixel correspondence 부족, inference 시 고정된 parameter 로 인해 유연하게 적응하지 못함
* CNN 의 단점을 해결하기 위해 <mark style='background:var(--mk-color-orange)'>self-attention mechanism</mark> 적용
	* 한 pixel 로부터 다른 모든 픽셀의 가중합을 계산
	* 이러한 self attention 을 사용하는 대표적인 모델이 transformer
	* <span style='color:var(--mk-color-teal)'>장점</span>: long-range pixel correspondence 연산 가능
	* <span style='color:var(--mk-color-red)'>단점</span>: resolution 이 올라갈수록 공간복잡도는 제곱으로 증가

* 이 논문에서는 global connectivity 를 유지하면서, large image 를 처리할 수 있는 모델을 제시: 
	* <mark style='background:var(--mk-color-red)'>Multi-head SA</mark> 대신 선형복잡도를 가지는 <mark style='background:var(--mk-color-green)'>multi-Dconv head transposed attention block(MDTA)</mark> 도입
		* image 의 pixel level 이 아닌, feature channel 간의 cross-covariance 를 input feature 에서 attention map 을 획득한다.
	* Gated Dconv Feed-forward Network(GDFN): 
		* 기존의 transformer 에서 attention 결과를 전달하는 역할
		* 기존 transformer 에서는 2개의 MLP layer 로 구성됨
		* 이 논문에서는 단순한 MLP layer 가 아닌 <mark style='background:var(--mk-color-green)'>gating mechanism 을 차용</mark>
			* 어떠한 complementary feature 가 forward 로 전달되어야 하는지를 control
![[Pasted image 20240518175558.png|500]]

# 2. Background

## Image restoration
* CNN 기반의 image restoration 은 encoder-decoder 구조인 U-Net 을 사용하였을 때 성능이 좋음
* 또한 skip connection 을 도입하였을 때도 효과적임

## Vision Transformers
* computer vision 분야에서 도입된 transformer: ViT
* 이는 여러 task 에 적용이 되었고 좋은 성능을 보였으나 image restoration 분야에서는 computational cost 문제로 도입이 어려웠음
* 직관적인 해결책은 local 영역에서 transformer 를 사용하는 Swin Transformer 를 사용하는 것.
* 하지만 이는 global connectivity 를 저해한다.
* 따라서 이 논문에서는 long-range dependency 를 유지하며 계산 효율성도 유지하는 방법을 제시

# Method
![[Pasted image 20240518205835.png]]
* 아래부터는 "overview", "multi-Dconv head transposed attention", "gated-Dconv feed-forward network" 순으로 설명한다.

## Overview Pipeline
* Degraded image($\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$) 를 input 으로 사용
* 3x3 conv 를 거쳐 low-level feature embedding $\mathbf{F}_0$ 추출: $\mathbf{F}_0 \in \mathbb{R}^{H \times W \times C}$
* 이 모델은 encoder-decoder 구조이다.
* 각 encoder-decoder 는 transformer block 을 포함한다.
	* 즉, downsamling-upsampling 과정을 거친다.
* downsampling, upsampling 은 각각 pixel-shuffle, pixel-unsuffle 을 사용한다.
* 

