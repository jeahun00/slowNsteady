
# Abstract
* dense prediction task 에서 CNN 대신 transformer 를 사용하는 dense vision transformer 를 소개한다.
* 이는 여러 단계에서 token 을 수집하여 image 와 같은 representation 을 구성하고, 이를 conv decoder 를 사용하여 전체 해상도의 예측으로 점진적으로 결합한다.

* dense vision transformer 는 cnn 에 비해 global 하게 일관된 정보를 제공
* 또한 대량의 학습데이터셋이 존재할 때 상당한 개선을 보여준다.

# 1. Introduction
* Dense prediction 을 위한 architecture
	* encoder: large scale dataset 으로 pretrained 된 image classification network(즉, backbone) 사용
	* decoder: encoder 의 특징을 집계하고 final dense prediction 으로 변환

* ==dense prediction 에 대한 구조적 연구는 주로 decoder 와 그 aggregation strategy 에 초점==을 맞춘다.
* 하지만 backbone 구조 자체가 모델 전체에 큰 영향을 미친다.
* 이는 잘못된 backbone 의 사용으로 encoder 에서 소실된 정보는 decoder 에서 회복할 수 없는 문제를 야기한다.

* ==CNN backbone 은 input image 를 convolution 을 통해 점진적으로 downsampling 하는 방법== 사용
	* <mark style='background:var(--mk-color-teal)'>장점</mark>: receptive field의 점진적 증가, high-level feature extract 등
	* <mark style='background:var(--mk-color-red)'>단점</mark>: 
		* deep cnn 의 경우 resolution 과 detail 이 손상
		* classification task 의 경우 큰 문제가 되지 않지만 <span style='color:var(--mk-color-red)'>pixel-level prediction 을 진행해야 하는 dense prediction task 에서는 치명적</span>
* 위와 같은 ==CNN 의 고질적인 단점을 해결하기 위해 아래 방법론 제시==
	* computation 자원이 허용하는 만큼의 높은 해상도에서 훈련
	* dilated convolution
	* skip connection
	* deep high-resolution representation(HR Net)
		* 이 링크 참고: [[🟢 Deep High-Resolution Representation Learning for Visual Recognition]]
* 하지만 <span style='color:var(--mk-color-red)'>convolution 연산의 특성상, 한번에 좁은 영역의 특성만을 볼 수 있기에 여전히 dense prediction 에서 cnn 의 적용은 한계</span>가 있다.

* 이에 이 논문에서는 <mark style='background:var(--mk-color-green)'>Dense Prediction Transformer(DPT)</mark> 를 소개한다.
	* ViT 를 encoder 의 backbone 으로 활용
	* ViT 특성상 초기 image embedding 추출 이후에서는 명시적인 downsampling 이 존재하지 않음
	* 또한 Transformer 의 특성으로 인해 global information 을 참조할 수 있다.
* 이러한 backbone 의 활용으로 monocular depth estimation, semantic segmentation 에서 좋은 성능을 이끌어냄


# 2. Related Work
* CNN 기반의 dense prediction
	* conv 로 downsample 이후 upsampling
* NLP 모델을 위한 transformer 구조
	* vision 을 위한 ViT

# 3. Architecture
* 기존의 dense prediction 에서 사용하던 encoder-decoder 구조는 유지
* 단, encoder 의 backbone 으로 ViT 사용
![[Pasted image 20240509193858.png]]

## Transformer encoder.
![[Pasted image 20240509202213.png|500]]
* transformer 에서는 bag-of-words(단어들의 순서는 고려하지 않고 빈도에 집중한 수치화된 표현) 사용
* ViT 에서는 단어들 대신, **image 에서 crop 된 patch** 들이 **단어의 역할을 대신**한다.
* CNN 에서는 네트워크를 통과하면 할수록 점진적으로 receptive field 가 커짐(네트워크를 통과할수록 global feature 를 참고할 수 있게 됨)
* 하지만 ViT 는 patch 의 해상도를 transformer 과정중 지속해서 유지하기에 모든 과정 중 global feature 를 참고할 수 있다.

* ViT 의 구조
	* $p^2$ 으로 이미지 crop
	* ResNet-50 으로 image embedding 추출
	* image patch 는 특정한 순서를 가지지 못함-> 따라서 positional embedding 을 추가
	* 또한, classification 을 위해 **readout token** 을 추가한다.

* $l$ 번째 transformer block 의 output: $t^l=\{t^l_0,...,t^l_{N_p}\}$
* classification 을 위한 readout token: $t_0$
* $N_p = \frac{HW}{p^2}$

* 이 논문에서는 3가지의 ViT 를 사용
	* 아래 3가지 작업에 대해 $p=16$ 으로 고정
	1. ViT-Base: 
		* Patch-based embedding(linear embedding 인가? -> 확인 필요)
		* 12개의 transformer layer
		* feature size $D=768$
	2. ViT-Large: 
		* Patch-based embeddings
		* 24개의 transformer layer
		* feature size $D=1024$
	3. ViT-Hybrid
		* ResNet-50 based embeddings
		* 12개의 transformer layer
		* feature size : input image 의 $\frac{1}{16}$ 에 해당하는 feature
* 위의 3가지 작업 모두 입력 patch 의 pixel 보다 더 큰 feature size 를 가지기에 아래 장점을 가짐
	* 기존의 정보 중 유익한 정보를 보존
	* 더욱 많은 정보를 반영할 수 있다.

## Convolution decoder

* convolution decoder 에서는 token 집합을 여러 해상도에서 image-likde feature representation 으로 assemble 한다.
* 이러한 feature representation 은 점진적으로 final dense prediction 으로 합쳐진다.
* transformer 의 임의의 계층의 출력 token 에서 image-like representation 을 복원하기 위해 3단계의 Reassemble operation 을 제안.


$$
\text{Reassemble}_s(t) = (\text{Resample} \circ \text{Concatenate} \circ \text{Read})(t)
$$
* $s$: input image 에 대한 회복된 representation 의 출력 크기 비율
* $\hat{D}$: output feature dimension
![[Pasted image 20240509203122.png|400]]
#### 1. Read stage

$$
\text{Read} : \mathbb{R}^{N_p+1 \times D} \rightarrow \mathbb{R}^{N_p \times D}
$$
* $Read$ 연산은 readout token 이 포함된 token 들과 output dimension 을 맞춰주는 역할
	* 이 과정을 통해 output 결과와 concat 을 가능하도록함

* Readout token 의 경우 dense prediction 작업에 명확한 목적을 제공하지는 않는다.
* 하지만 global information 을 포착하고 이를 각 token 에 분배하는데 도움을 줄 수 있다.
* $Read$ 연산은 아래 3가지 종류가 있다.
	1. $\text{Read}_{\text{ignore}}(t) = \{t_1, \dots, t_{N_p}\}$ 
		* 단순하게 readout token 을 무시하는 경우
	2. $\text{Read}_{\text{add}}(t) = \{t_1 + t_0, \dots, t_{N_p} + t_0\}$ 
		* readout token 을 다른 모든 token 에 골고루 더해주는 경우
	3. $\text{Read}_{\text{proj}}(t) = \{\text{mlp}(\text{cat}(t_1, t_0)), \dots, \text{mlp}(\text{cat}(t_{N_p}, t_0))\}$ 
		* readout token 과 일반 token 을 concat 한다.
		* linear layer->GELU 를 사용하여 원래 차원 $D$ 로 projection 한다.
#### 2. Concatenate stage
* $Read$ 과정 이후 $N_p$ 개 의 token 을 positional embedding 정보를 이용하여 원상태로 배치
* 이 과정을 통해 $\frac{H}{p} \times \frac{W}{p} \times D$ 의 feature representation 을 얻는다.
$$
\text{Concatenate} : \mathbb{R}^{N_p \times D} \rightarrow \mathbb{R}^{\frac{H}{p} \times \frac{W}{p} \times D}
$$

#### 3. Resample Stage
![[Pasted image 20240510111251.png|500]]
* $Concatenate$ 과정 이후 output size $\hat{D}$ 를 맞추기 위해 아래 과정을 거친다.
	* <mark style='background:var(--mk-color-purple)'>아마 여기서 말하는 output size 는 semantic segmentation 의 semantic map 의 channel 이나, depth estimation 의 depth map 의 channel 인 듯 한데 확인 필요</mark>
$$
\text{Resample}_s : \mathbb{R}^{\frac{H}{p} \times \frac{W}{p} \times D} \rightarrow \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times \hat{D}}
$$
* 1x1 conv 를 통해 input representation $D$ 를 $\hat{D}$ 로 projection
* $s\geq p$ : stride 가 있는 3x3 conv
* $s<p$ : stride 가 있는 3x3 transpose conv

![[Pasted image 20240510142857.png|500]]
* 위의 Fusion 과정은 3종류의 ViT backbone 에 따라 다르게 적용된다.
	1. ViT-Base: $l=\{3,6,9,12\}$
	2. ViT-Large: $l=\{5,12,18,24\}$
	3. ViT-Hybrid: $l=\{9,12\}$
* 위의 fusion 이후의 feature map 의 차원 $\hat{D}=256$ 으로 생성

## Handling varying image sizes
* DPT 는 여러 크기의 이미지를 다룰 수 있다.
* image size 가 달라지더라도 $p$ 로 나눌수만 있다면 처리가 가능하다.
* 하지만 positional embedding 은 위치 정보를 포함하기에 이미지의 크기에 영향을 받는다.
* 이는 bilinear interpolation 을 통해 적절한 크기로 변형함으로써 해결한다.

# 4. Experiment
* DPT 를 아래 2가지 dense prediction 에 적용한다.
	1. monocular depth estimation
	2. semantic segmentation
* 이 때, DPT 와 비슷한 용량을 가지는 convolution 과 비교하였을 때, large-scale dataset 을 적용하면 DPT 는 convolution 에 비해 더 큰 성능향상이 있다.
* 단안 깊이 추정 작업에서는 <mark style='background:var(--mk-color-teal)'>DPT</mark>가 기존의 최고 성능을 달성한 convolutional network에 비해 <span style='color:var(--mk-color-teal)'>깊이 예측의 정확도와 세밀도에서 현저한 개선</span>을 보여준다. 
* 의미론적 분할에서도 <mark style='background:var(--mk-color-teal)'>DPT</mark>는 특히 <span style='color:var(--mk-color-teal)'>복잡한 장면에서의 레이블 분할의 정확도를 향상시키는 데 큰 효과</span>를 나타냈다.


## 4.1. Monocular Depth Estimation
* monocular depth estimation 은 dense regression task 이다.

### Experiment protocol.
* Loss: [이 논문](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.pdf) 의 loss term 을 그대로 사용
(ZhengqiLiandNoahSnavely.MegaDepth:Learning single-view depth prediction from Internet photos. In CVPR, 2018.)
* Dataset:
	* 기존에 사용되던 dataset 들: MIX 5 라 지칭
	* MIX 6 : 5개의 추가 데이터셋을 포함한 meta dataset
		* 약 140만 개의 image
		* 현존하는 데이터셋 중 가장 큼
* Optimizer: Adam
* Learning rate: 
	* backbone: 1e-5
	* decoder weight: 1e-4
* crop
	* 이미지의 가장 긴 면이 384 pixel 이 되도록 크기를 조정
	* 384 x 384 크기의 이미지를 사용
* 72000 step, 16 batch size per 1 epoch, 60 epoch

### Zero-shot cross-dataset transfer
![[Pasted image 20240510150130.png]]
* Table 1 은 훈련 중에 보지 못한 다른 dataset 으로 zero-shot transfer 결과를 보여준다.
(evaluation metric 은 [이 논문](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9178977)에서 사용된 metric 을 사용하였는데 넣을지말지 고민중)
([30] Rene ́ Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. TPAMI, 2020.)
* 위의 6개의 metric 에 대해 모두 SOTA 달성(6개의 metric 모두 낮을수록 좋은 수치)
* DPT-Hybrid 의 경우 기존의 SOTA 와 비슷한 용량과 inference time 으로 SOTA 달성
* DPT-Large 의 경우 기존의 SOTA 에 비해 3배의 용량을 가지며, inference time 은 비슷
![[Pasted image 20240510150945.png|500]]
* 또한 이 논문에서는 DPT 가 large-scale dataset 때문만은 아닌지 판단하기 위해 MiDaS model 도 MIX6 에서 재훈련하여 평가를 진행하였다.
* large-scale dataset 은 MiDaS model 에 확실한 이점을 제공하지만 DPT 가 여전히 SOTA 성능이다.
![[Pasted image 20240510151322.png|500]]
* Figure 2 에서도 볼 수 있듯, DPT 는 MiDaS 이 비해 detail 을 더 잘 잡을 수 있다.

## 4.2. Semantic Segmentation

* semantic segmentation은 discrete labeling 작업을 대표하며, dense prediction 아키텍처를 검증하는 매우 경쟁적인 분야이다. 
* 이전 실험과 동일한 백본과 디코더 구조를 사용한다. 
* 이 논문에서는 half resolution에서 예측하고 bilinear interpolation을 사용하여 로짓을 전체 해상도로 업샘플링하는 출력 헤드를 사용한다. 
* 인코더는 다시 ImageNet에서 사전 훈련된 가중치로 초기화되고, 디코더는 무작위로 초기화된다.

### Experiment protocol.
* [이 논문](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf) 의 protocol 을 밀접하게 따른다
[51] Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li, and Alexander Smola. ResNeSt: Split- attention networks. arXiv preprint arXiv:2004.08955, 2020.
* loss: cross-entropy loss
* metric: pixACC(pixel 별 정확도), mIoU

### ADE20K
* DPT 를 해당 dataset 에 대해 240 epoch 동안 학습
* DPT-Hybrid 는 기존의 fully convolutional architecture 를 능가한다.
* DPT-Large 에서는 오히려 성능이 떨어진다.
	* 이는 DPT-Large 가 학습한 데이터셋보다 ADE20K 의 규모가 너무 적기에 발생한 문제이다.
![[Pasted image 20240510154013.png|500]]
![[Pasted image 20240510154106.png]]
* Figure 3에서도 알 수 있듯 DPT 가 더 세밀한 segment 결과를 보여준다.

## 4.3. Ablations
* Ablation sutdy 에서는 monocular depth estimation 을 task 로 잡는다.
* 기존의 MIX 6 dataset 이 아닌 3개의 dataset 에 대한 축소된 meta dataset 으로 ablation study 를 진행함
	* HRWSI
	* BlendedMVS
	* ReDWeb
* 위의 3개의 dataset 은 비교적 고품질의 정밀한 ground truth 를 제공하기에 채택되었다.
* 아래 experiment 에서 특별히 명시되지 않는 한, backbone 으로 ViT-Base 를 사용한다.

### Skip Connections.
![[Pasted image 20240510154844.png|500]]
* Base:
	* 결과에서도 볼 수 있듯, high-level feature 만 사용하는 것보다 low-level 과 high-level 을 같이 사용하는 것이 더 효과적
* Hybrid:
	* $R0,R1$: ResNet50 embedding network 의 첫번째 및 두 번째 downsampling 단계(layer)에서 추출된 feature 를 활용하는 것.
	* 이 부분 역시 high-level, low-level feature 를 동시에 활용하는 것이 더 좋은 성능을 낼 수 있음을 명시한다.

### Readout token
* Reassemble block 의 첫 단계에서 사용하는 readout token(for classification) 에 대한 연구
![[Pasted image 20240510155257.png|500]]
* token 을 무시하는 방식($Ignore$)이 일부 데이터셋에서는 좋은 성능을 보이지만 $Project$ 하는 것이 평균 성능은 더 좋다.

### Backbones.
![[Pasted image 20240510155507.png|500]]
* ViT-Large 를 사용하는 것이 성능은 가장 좋다(평균 값이 높음)
* 하지만 ViT-Large 는 ViT-Base 나 ViT-Hybrid 에 비해 3배의 parameter 를 가진다.
* 따라서 ViT-Hybrid 를 사용하는 것이 가장 balance 가 좋다.

### Inference resolution.
* convolution 의 경우 같은 모델을 사용하였을 때 input image 의 resolution 이 커지면 input image 에 대한 receptive field 의 비율이 작아진다.
* 이는 input image 의 resolution 이 커질수록 성능이 떨어지는 결과를 낳는다.
* 하지만 DPT 는 다른 모델에 비해서 resolution 이 커질수록 성능하락의 폭이 적다.
* 이는 DPT 가 resolution 에 덜 의존적이라는 것을 보여준다.
![[Pasted image 20240510161142.png|500]]

### Inference speed.
* DPT-Hybrid 와 DPT-Large 는 MiDaS 에서 사용하는 fully convolutional architecture 와 유사한 latency 를 보인다.
* DPT-Large 의 경우 다른 architecture 에 비해 높은 memory 를 요구하지만 높은 병렬성을 통해 다른 모델과 비슷한 latency 를 보인다.
