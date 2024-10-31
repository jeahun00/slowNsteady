# Abstract

참조 기반 초해상도(Reference-based Super-Resolution, RefSR)는 원격 센싱 이미지의 spatial 및 시간적 해상도를 연결하는 다리를 구축할 잠재력이 있습니다. 그러나 기존의 RefSR 방법은 큰 스케일 팩터에서 내용 재구성의 충실도와 texture 전이의 효과성에 한계가 있습니다. 조건부 확산 모델(Conditional Diffusion Models)은 현실적인 고해상도 이미지를 생성하는 새로운 기회를 열었지만, 이러한 모델 내에서 참조 이미지를 효과적으로 활용하는 것은 여전히 추가 연구가 필요한 분야입니다. 또한, 관련 참조 정보가 없는 영역에서는 내용 충실도를 보장하기 어렵습니다. 이러한 문제를 해결하기 위해, 우리는 RefSR을 위한 변화 인지 확산 모델(Change-aware Diffusion Model)인 Ref-Diff를 제안하며, denoising 과정을 명시적으로 안내하기 위해 land cover 변화(priors)를 사용합니다. 구체적으로, 변화하지 않은 영역에서 참조 정보를 활용하고 변화된 영역에서 semantic으로 관련된 내용을 재구성하기 위해 denoising 모델에 priors를 주입합니다. 이 강력한 가이드를 통해 semantic denoising와 참조 texture-aware denoising 과정을 분리하여 모델 성능을 향상시킵니다. 광범위한 실험을 통해 정량적 및 정성적 평가에서 최첨단 RefSR 방법에 비해 제안된 방법의 뛰어난 효과성과 견고성이 입증되었습니다. 코드와 데이터는 [GitHub 링크](https://github.com/dongrunmin/RefDiff)에서 확인할 수 있습니다.

# 1. Introduction
고해상도 원격 센싱 이미지의 시spatial 무결성은 정밀한 도시 관리, 장기간의 도시 개발 연구, 재해 모니터링 및 기타 원격 센싱 응용 프로그램에 필수적입니다 [7, 14, 50].

그러나 원격 센싱 기술의 한계와 높은 하드웨어 비용으로 인해 우리는 큰 규모에서 고해상도와 고시간 해상도를 동시에 달성할 수 없습니다 [25, 48]. 이 문제를 해결하기 위해 참조 기반 초해상도(Reference-based Super-Resolution, RefSR)는 지리적으로 쌍을 이루는 고해상도 참조(Reference, Ref) 이미지와 저해상도(Low-Resolution, LR) 이미지를 활용하여 다른 센서로부터 미세한 spatial 내용과 높은 재방문 빈도를 통합할 수 있습니다 [8]. 다양한 RefSR 방법들이 큰 진전을 이루었지만, 이 시나리오에서는 두 가지 주요 과제가 남아 있습니다.

첫 번째 과제는 Ref와 LR 이미지 간의 land cover 변화입니다. 자연 이미지 도메인과 달리 Ref 이미지는 이미지 검색을 통해 수집되거나 다른 관점에서 촬영된 반면, 원격 센싱 시나리오에서는 동일한 위치를 맞추기 위해 지리적 정보를 활용합니다. 기존 방법들은 적응 학습이나 attention-based transformers를 통해 LR과 Ref 이미지 간의 land cover 변화를 암묵적으로 포착합니다 [5, 27]. 그러나 이러한 방법들에서도 여전히 Ref 정보의 미사용 또는 오용 문제가 존재합니다.

두 번째 과제는 원격 센싱 센서 간의 큰 공간 해상도 격차(예: 8×에서 16×)입니다. 기존 RefSR 방법은 일반적으로 생성적 적대 신경망(Generative Adversarial Network, GAN)을 기반으로 하며, 4× 스케일 팩터에 맞게 설계되었습니다 [52, 55]. 이러한 방법들은 큰 스케일 팩터 초해상도에서는 세부 사항을 거의 재구성하거나 전송할 수 없습니다. 최근 몇 년간 조건부 확산 모델(Conditional Diffusion Models)은 이미지 초해상도와 재구성에서 GAN보다 더 큰 효과를 보여주었습니다 [12, 42]. RefSR을 향상시키는 간단한 방법은 확산 모델의 조건으로 LR 및 Ref 이미지를 사용하는 것입니다. 참조 정보를 효과적으로 활용하기 위해, 일부 방법들은 denoising 네트워크 블록에 Ref 정보를 주입합니다 [19, 39]. 그러나 이러한 방법들은 denoising를 위해 LR 및 Ref 이미지 간의 관계를 암묵적으로 모델링하여 Ref 정보의 모호한 사용과 내용 충실도 제한으로 이어집니다.

위의 문제를 완화하기 위해, 우리는 참조 특징 사용의 효과성과 내용 재구성의 충실도를 향상시키기 위해 land cover 변화 priors를 도입합니다 (그림 1 참조). 원격 센싱 변화 감지(Change Detection, CD)의 발전 덕분에, 우리는 서로 다른 공간 해상도의 이미지 간 land cover 변화를 효과적으로 포착하기 위해 상용 CD 방법을 사용할 수 있습니다 [23, 33, 53]. 한편으로는 land cover 변화 priors가 변하지 않은 영역에서 참조 정보의 활용을 강화합니다. 다른 한편으로는 변경된 land cover 클래스가 변경된 영역에서 semantic으로 관련된 내용의 재구성을 안내할 수 있습니다. 또한, land cover 변화 priors에 따라 semantic denoising와 참조 texture-aware denoising를 반복적인 방식으로 분리하여 모델 성능을 향상시킬 수 있습니다. 제안된 방법의 효과를 입증하기 위해, 두 개의 대규모 스케일 팩터를 사용한 두 데이터셋에서 실험을 수행하였습니다. 우리의 방법은 최첨단 성능을 달성합니다. 요약하자면, 우리의 기여는 다음과 같습니다:

- 우리는 RefSR에서 land cover 변화 priors를 도입하여, 변하지 않은 영역에서의 texture 전이 효과성과 변경된 영역에서의 재구성 충실도를 향상시켜 원격 센싱 시나리오에서 spatial 및 시간적 해상도를 연결합니다.
- 우리는 land cover 변화 priors를 조건부 확산 모델에 주입하여 변화 인지 denoising 모델에 의해 Ref-Diff라는 새로운 RefSR 방법을 제안합니다. 이로써 큰 스케일 팩터 초해상도에서 모델의 효과성을 향상시킵니다.
- 실험 결과는 제안된 방법이 정량적 및 정성적 측면에서 기존의 SOTA RefSR 방법보다 뛰어나다는 것을 보여줍니다.

# 2. Related Works
## 2.1 Reference-Based Super-Resolution Methods

단일 이미지 초해상도(Single-Image Super-Resolution, SISR)와 비교할 때, 참조 기반 초해상도(Reference-based Super-Resolution, RefSR)는 잘못된 문제를 완화하고 현실적인 텍스처를 복원하는 데 큰 잠재력을 보여줍니다 [1, 24]. 구체적으로, Jiang et al. [15]은 LR 및 Ref 이미지 간의 텍스처 전이 및 해상도 격차 문제를 해결하기 위해 대조적 대응 네트워크와 교사-학생 상관 관계 증류 방법을 제안합니다. RRSR [49] 및 AMSA [41]는 고품질 대응 매칭에 기여합니다. 또한, Huang et al. [13]은 초해상도 및 텍스처 전이 작업을 분리하여 Ref 이미지의 미사용 및 오용 문제를 완화합니다.

지리적 위치를 통해 LR 및 Ref 이미지를 사전 매칭하기 때문에, 기존 원격 센싱 이미지용 RefSR 방법들은 관련 텍스처를 변환하고 관련 없는 정보 융합을 억제하는 것을 목표로 합니다 [5, 47]. 그러나 그 결과는 변경된 영역과 변경되지 않은 영역 간의 내부 해상도 불일치가 명확하게 나타납니다. 변경된 영역의 세부 사항은 GAN 기반 방법을 사용하여 거의 재구성할 수 없기 때문입니다. 따라서 최근 연구들은 더 현실적인 결과를 생성하기 위해 확산 모델을 채택합니다 [19]. 예를 들어, HSR-Diff [39]는 조건부 확산 모델을 적용하고 교차 주의 메커니즘을 조건화 메커니즘으로 활용하여 denoising 과정에 LR 및 Ref 특징을 통합하여 지각적 품질을 향상시킵니다. 그러나 LR 및 Ref 이미지 간의 명시적 관계 모델링이 부족하여 denoising 과정의 어려움과 결과의 불확실성이 증가합니다. 이 연구에서는 land cover 변화 priors를 도입하고 이를 명시적으로 사용하여 denoising 과정을 안내합니다.

## 2.2 Conditional Diffusion Model for Super-Resolution

확산 모델의 이점을 통해 최근 이미지 초해상도 기술은 시각적 매력과 고품질 출력 면에서 상당한 진전을 이루었습니다. 초기 연구들 [18, 31]은 큰 스케일 팩터 초해상도를 처리하기 위해 LR 이미지를 조건으로 사용하여 확산 과정을 다루었습니다. 이미지 초해상도의 효과를 더욱 향상시키기 위해, 일부 연구들은 denoising 과정을 안내하기 위해 향상된 조건을 탐구합니다. 예를 들어, ResDiff [32] 및 ACDMSR [28]는 생성 과정을 가속화하고 우수한 샘플 품질을 확보하기 위해 CNN으로 향상된 LR 예측을 조건으로 사용합니다. BlindSRSNF [40] 및 Dual-Diffusion [43]은 열화 표현을 확산 모델의 조건으로 결합하여 실제 시나리오에서 만족스러운 결과를 얻습니다.

이러한 priors를 조건부 확산 모델의 입력과 단순히 결합하는 것 외에도, 최근 연구들은 이를 denoising 모델에 통합합니다 [9, 35]. Wang et al. [36]은 다층 공간 적응 정규화 연산자를 통해 semantic 이미지 합성을 위해 디코더에 semantic 레이아웃을 도입합니다. PASD [46] 및 DiffBIR [21]은 CLIP에서 추출된 고수준 정보를 또는 열화 제거로 향상된 LR 표현과 같은 priors를 도입하기 위해 ControlNet을 채택합니다. 본 연구에서는 RefSR에서 land cover 변화 priors의 활용을 탐구합니다. 우리는 denoising 모델에 priors를 주입하여 semantic denoising와 참조 texture-aware denoising를 분리합니다.

## 2.3 Change Detection

변화 감지(Change Detection, CD) 방법의 발전으로, 기존 연구들은 서로 다른 land cover 카테고리에서 최대 80%의 F1-score를 달성할 수 있습니다 [26]. 실용적인 응용을 위해, 최근 CD 모델들은 경량화되는 경향이 있으며 서로 다른 해상도의 다중 시점 이미지를 처리할 수 있습니다 [23]. 예를 들어, Zheng et al. [53]은 두 시점 이미지 간의 해상도 격차를 연결하기 위해 크기 조정 작업 없이 교차 해상도 차이 학습을 설계하였습니다. Liu et al. [22]은 쌓아 올린 attention 모듈을 갖춘 SISR 기반 변화 감지 네트워크를 제안하여 건물 CD 작업에서 8× 해상도 차이에서 83% 이상의 F1-score를 달성했습니다. 이러한 CD 방법들은 높은 신뢰성과 플러그 앤 플레이 기능을 보여주며, 이 연구를 위해 고품질의 land cover 변화 prior 정보를 직접 제공할 수 있습니다.

# 3. Methodology

이 논문에서는 큰 스케일 팩터에서 RefSR 방법의 효과를 높이기 위해 조건부 확산 모델(Conditional Diffusion Model)을 채택합니다. 생성된 내용의 충실도를 향상시키고 Ref 이미지 전이의 효과성을 개선하기 위해, 우리는 land cover 변화 priors를 도입합니다. 제안된 방법의 아키텍처는 그림 2에 나와 있습니다. 우리는 denoising 블록에 Ref 특징과 land cover 변화 priors를 주입하는 새로운 변화 인지 denoising 모델을 제안합니다. 이러한 priors를 활용하여, 우리는 디코더에서 semantic denoising와 참조 texture-aware denoising 과정을 분리하고 두 과정을 반복적으로 처리합니다.

## 3.1 Preliminary

조건부 확산 모델(Conditional Diffusion Model)은 기본 확산 모델을 확장하여 조건을 포함하고, 전방 및 역방향 확산 과정을 통합합니다. Karras et al. [16]은 다양한 역방향 확산 모델을 EDM 프레임워크로 통합합니다. EDM의 훈련 목표는 다음과 같이 정의됩니다:

$$
\mathbb{E}_{\sigma, \mathbf{y}, \mathbf{n}} [\lambda(\sigma) \| D(\mathbf{y} + \mathbf{n}; \sigma) - \mathbf{y} \|_2^2],
$$

여기서 표준 편차 $\sigma$는 노이즈 레벨을 제어하고, $\mathbf{y}$는 훈련 이미지이며 $\mathbf{n}$은 노이즈입니다. $D(\cdot)$는 denoising 함수이고, $\lambda(\sigma)$는 손실 가중치입니다.

신경망을 효과적으로 훈련시키기 위해 EDM의 사전 조건화는 다음과 같이 정의됩니다:

$$
D_\theta(\mathbf{x}; \sigma) = c_{\text{skip}}(\sigma)\mathbf{x} + c_{\text{out}}(\sigma) F_\theta (c_{\text{in}}(\sigma)\mathbf{x}; c_{\text{noise}}(\sigma)),
$$

여기서 $\mathbf{x} = \mathbf{y} + \mathbf{n}$입니다. $F_\theta(\cdot)$는 훈련 중인 신경망을 나타냅니다. $c_{\text{skip}}$는 skip 연결을 조정하고, $c_{\text{in}}$ 및 $c_{\text{out}}$은 입력 및 출력 크기를 각각 조정합니다. $c_{\text{noise}}$는 노이즈 레벨 $\sigma$를 훈련 신경망의 조건 입력으로 매핑하는 데 사용됩니다.

이 연구에서는, 확산 아키텍처가 EDM에서의 훈련 목표, 사전 조건화 및 기타 구현들을 따릅니다.

## 3.2 Change-Aware Denoising Model

이 연구는 원격 센싱 이미지를 위한 큰 스케일 팩터에서 RefSR을 촉진하기 위해 land cover 변화 priors를 활용하는 것을 목표로 합니다. 제안된 변화 인지 denoising 모델은 그림 2(a)에 나와 있습니다. [29, 36]의 영감을 받아, land cover 변화 prior는 semantic denoising를 위해 LR 및 Ref 이미지 간의 변경된 영역의 semantic 레이아웃으로 간주될 수 있습니다. 한편, 참조 texture-aware denoising를 통해 변경되지 않은 영역의 texture 세부 사항을 향상시킬 수 있습니다. 결과적으로, semantic denoising와 참조 texture-aware denoising 과정은 change-aware decoder에서 분리될 수 있으며(그림 2(c) 참조), 이는 denoising 결과를 더욱 향상시킵니다.

### Land Cover Change Priors
이 연구에서 사용된 land cover 변화 priors는 각 이미지 쌍에 대한 픽셀 수준의 다중 카테고리 변화 감지 마스크로, 변경 없음 클래스와 다양한 land cover 변화 클래스를 포함합니다. 훈련에서 land cover priors의 잠재력을 완전히 발휘하기 위해, 우리는 조건으로 land cover 변화 마스크의 실제 값을 사용합니다. 실제 응용에서는 기성의 종단간 변화 감지 방법 또는 두 단계의 land cover 분류 방법으로 변화 감지 마스크를 생성할 수 있습니다. 그림 2(a)에 나와 있듯이, land cover 변화 마스크는 노이즈와 함께 입력으로 결합되며, change-aware decoder에 주입됩니다.

### Change-Aware Encoder
계산 성능을 향상시키고 LR denoising의 과도한 개입을 피하기 위해, LR 이미지, Ref 이미지, land cover image 및 land cover 변화 마스크는 노이즈 있는 이미지와 연결되어 인코더의 입력으로 사용되며, [39]처럼 인코더 블록에 주입되지 않습니다. 인코더의 아키텍처는 [16]에서 개선된 U-Net을 기반으로 합니다(그림 2(b) 참조). 각 change-aware encoder 블록은 그룹 정규화, 컨볼루션, SiLU 및 다중 헤드 attention 모듈로 구성됩니다. 각 시점 $t$는 특정 노이즈 레벨에 해당하므로, 시점 임베딩을 학습 가능한 가중치 $w(t)$와 특징을 조절하는 바이어스 $b(t)$에 매핑합니다. 다중 헤드 attention은 각각 자체 학습 가능한 매개변수 세트를 가진 상태로, 병렬로 여러 번의 attention 과정을 거칩니다 [34].

### Change-Aware Decoder

그림 2(c)에 나와 있듯이, 우리는 change-aware decoder 블록에 land cover 변화 마스크와 Ref 이미지의 특징을 주입합니다. land cover 변화 priors를 사용하여, 우리는 변경된 영역에서 semantic denoising와 변경되지 않은 영역에서 참조 texture-aware denoising를 분리합니다. land cover 변화 priors의 오라벨 문제를 해결하기 위해, 우리는 semantic spatial 특징 변환(Spatial Feature Transform, SFT) 모듈을 위해 land cover 변화 마스크와 LR 이미지를 결합합니다. land cover 변화 마스크의 정확도를 고려할 때, 변화 감지 방법을 통해 예측된 변화 마스크의 정확도는 실제 응용에서 보통 60%에서 80% 사이입니다. 우리는 원래의 SFT [37]와 SPADE [29, 36]처럼 안내 특징만 사용하는 대신, 안내 특징과 denoising 특징을 결합하여 공간 적응 가중치와 바이어스를 학습합니다. 수정된 SFT 모듈은 다음과 같이 공식화할 수 있습니다:

$$
F_{i+1} = \gamma_i (F_e \oplus F_i) \cdot F_i + \beta_i (F_e \oplus F_i),
$$

여기서 $F_i$와 $F_{i+1}$는 각각 SFT 모듈의 입력 및 출력 특징입니다. $\gamma_i(\cdot)$와 $\beta_i(\cdot)$는 추출기에서 얻은 안내 특징 $F_e$와 입력 특징 $F_i$의 결합으로부터 학습된 공간 적응 가중치와 바이어스입니다.

## 3.3 Degradation Model and Implementation Details

우리는 실세계 시나리오에서의 훈련을 위해 LR 이미지를 시뮬레이션하기 위해 종합적인 열화를 채택합니다. 상용 블라인드 초해상도(Blind Super-Resolution) 방법 [10, 38]과 원격 센싱 센서의 특성 [6, 30]에 따라, 등방성 가우시안 블러, 비등방성 가우시안 블러, 모션 블러, 다양한 보간 방법을 사용한 리사이즈, 가산 가우시안 노이즈, JPEG 압축 노이즈를 사용하여 합성 LR 이미지를 생성합니다. 열화 복잡도의 설정은 스케일 팩터에 기반합니다. 실험에서, 16× 데이터셋을 위한 열화 모델은 8× 데이터셋을 위한 것보다 단순합니다.

훈련 중에는, 각 고해상도(HR) 이미지, Ref 이미지, 및 land cover 변화 마스크가 256 × 256 크기로 무작위로 크롭되고, 해당 LR 이미지의 크기는 스케일 팩터와 관련이 있습니다. 확산 모델의 구현은 [16]에 따릅니다. 우리는 드롭아웃 비율을 0.2로 설정하고, 배치 크기를 48로 설정합니다. Adam 옵티마이저를 사용하며, $\beta_1 = 0.9$ 및 $\beta_2 = 0.999$입니다. 학습률은 $1 \times 10^{-4}$로 초기화됩니다. 모델은 4개의 NVIDIA A800 GPU를 사용하여 500k 반복 횟수 동안 업데이트됩니다.



§
