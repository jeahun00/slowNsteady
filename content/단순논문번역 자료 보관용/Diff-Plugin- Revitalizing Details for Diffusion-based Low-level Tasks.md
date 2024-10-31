
# Abstract

대규모 데이터셋으로 학습된 Diffusion models는 image synthesis에서 놀라운 성과를 달성하였습니다. 그러나, diffusion 과정에서의 무작위성으로 인해, 종종 세부 사항 보존이 필요한 다양한 low-level tasks를 처리하는 데 어려움을 겪습니다. 이러한 한계를 극복하기 위해, 단일 사전 학습된 diffusion model이 다양한 low-level tasks에서 고해상도의 결과를 생성할 수 있도록 하는 새로운 Diff-Plugin 프레임워크를 제시합니다. 구체적으로, 우리는 dual branch 설계를 갖춘 경량 Task-Plugin 모듈을 처음으로 제안하여, diffusion 과정에서 image content를 보존하는 데 task-specific priors를 제공합니다. 그 다음, 우리는 text instruction에 기반하여 다양한 Task-Plugins을 자동으로 선택할 수 있는 Plugin-Selector를 제안하여, 사용자가 natural language로 여러 low-level tasks를 지정함으로써 이미지를 편집할 수 있게 합니다. 우리는 8개의 low-level vision tasks에서 광범위한 실험을 수행합니다. 결과는 특히 real-world scenarios에서 기존 방법들에 비해 Diff-Plugin의 우수성을 입증합니다. 우리의 ablations는 Diff-Plugin이 안정적이고, 일정 조정이 가능하며, 다양한 dataset 크기에 걸쳐 견고한 학습을 지원함을 추가로 검증합니다. 프로젝트 페이지: [https://yuhaoliu7456.github.io/Diff-Plugin](https://yuhaoliu7456.github.io/Diff-Plugin)

# 1. Introduction

지난 2년 동안, diffusion models \[9, 21, 22, 61\]은 이미지 생성에서 전례 없는 성공을 거두었고, vision foundation models로서의 잠재력을 보여주었습니다. 최근 많은 연구들 \[4, 25, 28, 31, 46, 91, 96\]은 대규모 text-to-image 데이터셋으로 학습된 diffusion models가 다양한 시각적 속성을 이미 이해할 수 있으며, 하위 작업들, 예를 들어 이미지 분류 \[31\], 세그멘테이션 \[25, 96\], 번역 \[46, 91\], 그리고 편집 \[4, 28\]을 위한 다재다능한 시각적 표현을 제공할 수 있음을 입증하였습니다.

그러나 diffusion 과정에서의 고유한 무작위성으로 인해, 기존 diffusion models는 입력 이미지의 일관된 내용을 유지하지 못하고 low-level vision tasks를 처리하는 데 실패합니다. 이를 해결하기 위해, 일부 방법들 \[46, 63\]은 DDIM Inversion \[61\] 전략을 통해 이미지를 편집할 때 입력 이미지를 prior로 활용할 것을 제안하지만, 장면이 복잡할 때에는 불안정합니다. 다른 방법들 \[16, 52, 56, 71, 83\]은 task-specific 데이터셋에서 새로운 diffusion models를 처음부터 학습하려고 시도하지만, 이는 단일 작업만 해결하는 데 한정됩니다.

본 연구에서는 task의 목표를 설명하는 정확한 text prompt가 이미 사전 학습된 diffusion model이 많은 low-level tasks를 해결하도록 지시할 수 있음을 관찰하였습니다. 그러나 이는 Fig. 2에서 설명된 것처럼 명백한 콘텐츠 왜곡을 초래합니다. 이 문제에 대한 우리의 통찰은 task-specific priors가 task의 안내 정보와 입력 이미지의 공간적 정보를 모두 포함하여 사전 학습된 diffusion models가 low-level tasks를 처리하면서 고해상도 콘텐츠 일관성을 유지하도록 충분히 안내할 수 있다는 것입니다. 이러한 잠재력을 활용하기 위해, 우리는 사전 학습된 diffusion model, 예를 들어 stable diffusion \[54\]이 다양한 low-level tasks를 처리하면서 원래의 생성 능력을 손상시키지 않도록 하는 첫 번째 프레임워크인 Diff-Plugin을 제안합니다.

**Diff-Plugin**은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, task-specific priors를 추출하는 데 도움이 되는 경량 Task-Plugin 모듈을 포함합니다. Task-Plugin은 Task-Prompt Branch (TPB)와 Spatial Complement Branch (SCB)로 이분화됩니다. TPB는 task guidance prior를 증류하여, diffusion model이 지정된 vision task를 향하도록 하고 복잡한 텍스트 설명에 대한 의존도를 최소화합니다. SCB는 TPB의 task-specific visual guidance를 활용하여 공간적 세부 사항의 캡처 및 보완을 돕고, 생성된 콘텐츠의 충실도를 향상시킵니다. 둘째, 다양한 Task-Plugins의 사용을 용이하게 하기 위해 Diff-Plugin은 사용자가 텍스트 입력을 통해 원하는 Task-Plugins을 선택할 수 있도록 하는 Plugin-Selector를 포함합니다 (시각적 설명은 Fig. 1에 나타나 있습니다). Plugin-Selector를 학습하기 위해, 우리는 task-specific visual guidance를 pseudo-labels로 사용하는 multi-task contrastive learning \[49\]을 적용합니다. 이는 Plugin-Selector가 다양한 visual embeddings을 task-specific text inputs과 일치시켜, Plugin-Selector의 견고성과 사용 편의성을 강화합니다.

우리의 방법을 철저히 평가하기 위해, 우리는 8개의 다양한 low-level vision tasks에서 광범위한 실험을 수행했습니다. 결과는 Diff-Plugin이 다양한 작업에서 안정적일 뿐만 아니라 눈에 띄는 스케줄링 가능성을 보여주어, 텍스트 기반의 multi-task 응용 프로그램을 용이하게 함을 확인했습니다. 추가로, Diff-Plugin은 다양한 크기의 데이터셋에 걸쳐, 500개 이하에서 50,000개 이상의 샘플로 확장하여, 기존의 학습된 플러그인에 영향을 미치지 않고도 그 확장성을 보여주었습니다. 마지막으로, 우리의 결과는 제안된 프레임워크가 기존 diffusion 기반 방법들에 비해 시각적, 정량적으로 우수하며, regression 기반 방법에 비해 경쟁력 있는 성능을 달성함을 보여줍니다.

우리의 주요 기여는 다음과 같이 요약됩니다:

- 우리는 사전 학습된 diffusion model이 원래의 생성 능력을 유지하면서 다양한 low-level tasks를 수행할 수 있도록 하는 첫 번째 프레임워크인 **Diff-Plugin**을 제시합니다.
- 우리는 결과의 충실도를 향상시키기 위해 diffusion 과정에 task-specific priors를 주입하도록 설계된 경량 dual-branch 모듈인 Task-Plugin을 제안합니다.
- 우리는 사용자가 제공한 텍스트를 기반으로 적절한 Task-Plugin을 선택하는 Plugin-Selector를 제안합니다. 이는 low-level vision tasks를 위한 텍스트 지시를 통해 이미지를 편집할 수 있는 새로운 응용 프로그램으로 확장됩니다.
- 우리는 8가지 작업에서 광범위한 실험을 수행하여, 기존의 diffusion 및 regression 기반 방법에 비해 Diff-Plugin의 경쟁력 있는 성능을 입증합니다.

# 2. Related Works

**Diffusion models** \[60, 62\]은 image synthesis \[9, 21, 22, 61\]에 적용되어 놀라운 성공을 거두었습니다. 광범위한 text-image data \[59\]와 대규모 language models \[49, 50\]을 통해, diffusion-based text-guided image synthesis \[2, 42, 51, 54, 57\]은 더욱 매력적으로 변모하였습니다. text-guided synthesis diffusion model을 활용하여, 여러 접근법은 텍스트 기반 편집을 위한 생성 능력을 활용합니다. Zero-shot approaches \[19, 46, 63\]은 올바른 initial noise \[61\]에 의존하고 주어진 위치에서 지정된 콘텐츠를 편집하기 위해 attention map을 조작합니다. Tuning-based strategies는 최적화된 DDIM inversion \[65\], attention tuning \[29\], text-image coupling \[28, 55, 93\] 및 prompt tuning \[10, 14, 39\]을 통해 이미지 충실도와 생성된 다양성 사이의 균형을 맞추려고 합니다. 반대로, InstructP2P \[4, 89\]는 latent diffusion \[54\]과 prompt-to-prompt \[19\]를 통해 훈련 및 편집을 위해 paired data를 생성합니다. 그러나 diffusion 과정에서의 무작위성과 task-specific priors의 부재는 세부 사항 보존이 필요한 low-level vision tasks에 있어 이들 방법을 불가능하게 만듭니다.

**Conditional generative models**은 조건과의 일관성을 보장하기 위해 다양한 외부 입력을 사용합니다. Training-free methods \[8, 76\]는 attention layers를 조작하여 지정된 위치에서 새로운 콘텐츠를 생성할 수 있지만, 조건 유형이 제한적입니다. Fine-tuning-based approaches는 새로운 diffusion branch \[40, 90, 94\] 또는 전체 모델 \[1\]을 훈련하여 사전 학습된 diffusion models에 추가적인 지시를 주입합니다. 이러한 방법들은 전반적인 구조적 일관성에도 불구하고, 출력과 입력 이미지 세부 사항 간의 고충실도를 보장할 수 없습니다.

**Diffusion-based low-level methods**는 zero-shot과 training-based로 구분될 수 있습니다. 전자는 사전 학습된 denoising diffusion-based generative models \[22\]에서 생성 priors를 빌려 linear \[27, 70\] 및/또는 non-linear \[7, 12\] image restoration tasks를 해결하지만, 실제 데이터에서는 종종 좋지 않은 결과를 낳습니다. 후자는 super-resolution \[58, 74\], JPEG compression \[56\], deblurring \[52, 73\], face restoration \[71, 95\], low-light enhancement \[24, 83, 92\], and shadow removal \[16\]과 같은 task-dependent designs를 통해 개별 모델을 학습하거나 미세 조정합니다. 동시 작업으로는 StableSR \[66\]과 DiffBIR \[34\]이 있으며, 열화된 또는 복원된 이미지를 사용하여 blind face restoration을 위해 diffusion models를 학습하는 조건부 diffusion branch를 사용합니다. 반면에, 우리의 프레임워크는 사전 학습된 diffusion model이 경량의 task-specific plugins을 통해 다양한 low-level tasks를 처리할 수 있도록 합니다.

**Multi-task models**은 다양한 작업, 예를 들어 object detection과 segmentation \[18\], rain detection and removal \[80\], adverse weather restoration \[45, 82, 98\], and blind image restoration \[33, 47\]에 걸쳐 상호 보완적인 정보를 학습할 수 있습니다. 그러나 이러한 방법들은 훈련 후에 정의된 작업만 처리할 수 있습니다. 대신에, 우리의 **Diff-Plugin**은 유연하여 task-specific plugins을 통해 새로운 작업을 통합할 수 있으며, 개별적으로 학습된 Task-Plugins을 사용합니다. 따라서 Diff-Plugin에 새로운 low-level tasks를 추가할 때, 우리는 사전 학습된 Task-Plugins을 프레임워크에 추가하기만 하면 되며, 기존 작업을 다시 학습할 필요는 없습니다.

# 3. Methodologies

In this section, we first review the diffusion model formulations (Sec. 3.1). Then, we introduce our **Diff-Plugin** framework (Sec. 3.2), which developed from our newly proposed Task-Plugin (Sec. 3.3) and Plugin-Selector (Sec. 3.4).

## 3.1. Preliminaries

Diffusion model은 forward process와 reverse process로 구성됩니다. Forward process에서, 깨끗한 입력 이미지 $\mathbf{x}_0$를 주어지면, diffusion model은 점진적으로 Gaussian noise를 추가하여 시간 단계 $t \in \{0, 1, \ldots, T\}$에서 noisy image $\mathbf{x}_t$를 생성합니다. 이는 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t$로 표현되며, 여기서 $\bar{\alpha}_t$는 사전 정의된 스케줄링 변수이고 $\epsilon_t \sim \mathcal{N}(0, \mathbf{I})$는 추가된 노이즈입니다. Reverse process에서, diffusion model은 표준 Gaussian noise $\mathbf{x}_T$로부터 노이즈를 반복적으로 제거하고, 최종적으로 깨끗한 이미지 $\mathbf{x}_0$를 추정합니다. 이는 일반적으로 노이즈 예측 네트워크 $\epsilon_\theta$를 학습시키는 데 사용되며, 노이즈 $\epsilon_t$에 의해 제공된 감독을 받습니다. 다음과 같이 수식으로 나타낼 수 있습니다:


$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, t, \epsilon \sim \mathcal{N}(0,1)} \left[ \left\| \epsilon - \epsilon_\theta (\mathbf{x}_t, t) \right\|_2^2 \right].$$



## 3.2. Diff-Plugin

우리의 주요 관찰은 ==사전 학습된 diffusion models의 고유한 zero-shot 능력==입니다. 이는 ==명시적인 task-specific 학습 없이 다양한 시각적 콘텐츠를 생성할 수 있게 하여 low-level vision tasks를 수행==할 수 있게 합니다. 그러나 이러한 능력은 보다 미세한 task-specific editing에서는 한계에 직면합니다. 예를 들어, 눈 제거 작업에서는 모델이 눈만 제거하고 다른 내용은 그대로 유지해야 하지만, Fig. 2에서 보이는 것처럼 diffusion 과정의 고유한 무작위성으로 인해 장면에서 눈 제거 외에 의도치 않은 변경이 발생하는 경우가 자주 있습니다. 이러한 불일치는 low-level vision tasks에서 세부 사항 보존을 위해 중요한 task-specific priors의 부족에서 비롯됩니다.

NLP \[75, 77\]와 GPT-4 \[43\]의 모듈형 확장에 영감을 받아, 우리는 대형 언어 모델의 핵심 역량을 손상시키지 않으면서 하위 작업을 강화하는 plug-and-play 도구를 활용하여, 유사한 아이디어를 기반으로 새로운 프레임워크인 **Diff-Plugin**을 소개합니다. 이 프레임워크는 여러 경량 플러그인 모듈인 Task-Plugin을 사전 학습된 diffusion models에 통합하여 다양한 low-level tasks를 처리합니다. Task-Plugins는 필수적인 task-specific priors를 제공하여 모델이 고충실도와 task-consistent 콘텐츠를 생성할 수 있도록 안내합니다. 또한, diffusion models는 특정 시나리오에 대한 text instructions를 기반으로 콘텐츠를 생성할 수 있지만, 다양한 low-level tasks에 대한 Task-Plugins을 스케줄링할 수 있는 능력이 부족합니다. 기존의 conditional generation methods \[48, 90\]는 입력 조건 이미지를 통해 다른 생성 작업을 지정할 수 있을 뿐입니다. 따라서, 원활한 텍스트 기반 작업 스케줄링을 촉진하고 복잡한 워크플로우에서 다양한 Task-Plugins 간 전환을 가능하게 하기 위해, **Diff-Plugin**은 사용자가 텍스트 명령을 통해 적절한 Task-Plugins을 선택하고 스케줄링할 수 있도록 Plugin-Selector를 포함합니다.

Fig. 3은 **Diff-Plugin** 프레임워크를 묘사합니다. 주어진 이미지에서, 사용자는 텍스트 프롬프트를 통해 작업을 지정하고, Plugin-Selector는 이를 위해 적절한 Task-Plugin을 식별합니다. Task-Plugin은 그런 다음 이미지를 처리하여 task-specific priors를 추출하고, 사전 학습된 diffusion model이 사용자가 원하는 결과를 생성할 수 있도록 안내합니다. 단일 플러그인의 범위를 넘어서는 더 복잡한 작업의 경우, **Diff-Plugin**은 미리 정의된 매핑 테이블을 사용하여 작업을 하위 작업으로 분해합니다. 각 하위 작업은 지정된 Task-Plugin에 의해 처리되며, 프레임워크가 다양한 복잡한 사용자 요구 사항을 처리할 수 있는 능력을 보여줍니다.

## 3.3. Task-Plugin

Fig. 4에 설명된 것처럼, 우리 Task-Plugin 모듈은 두 개의 분기로 구성됩니다: Task-Prompt Branch (TPB)와 Spatial Complement Branch (SCB). TPB는 text-conditional image synthesis \[54\]에서의 텍스트 프롬프트 사용과 유사하게 사전 학습된 diffusion model에 task-specific guidance를 제공하는 데 중요합니다. 우리는 사전 학습된 CLIP vision encoder \[49\]를 통해 추출한 visual prompts를 사용하여 모델의 초점을 task와 관련된 패턴(예: 비 제거를 위한 비 줄무늬나 눈 제거를 위한 눈송이)으로 맞춥니다. 구체적으로, 입력 이미지 $I$에 대해, 인코더 $Enc_I(\cdot)$는 먼저 일반적인 시각적 특징을 추출하고, 이를 TPB가 변별적인 시각적 가이드 priors $F^p$로 증류합니다:


$$F^p = TPB(Enc_I(I)),$$


여기서 TPB는 Layer Normalization 및 LeakyReLU 활성화(최종 레이어 제외)를 갖춘 세 개의 MLP 레이어로 구성되어 가장 task-specific 속성만을 유지합니다. 이 접근법은 $F^p$를 모델이 텍스트 기반 생성 과정에서 일반적으로 사용하는 텍스트 특징과 일치시켜, Plugin-Selector에 대한 더 나은 작업 정렬을 촉진합니다. 또한, visual prompts 사용은 복잡한 텍스트 프롬프트 엔지니어링이 필요 없는 사용자의 역할을 단순화하여, 이는 종종 특정 vision tasks에 대해 도전적이고 미세한 텍스트 변이에 민감합니다 \[78\].

그러나 task-specific visual guidance prior $F^p$는 전역 의미적 속성을 유도하는 데 중요하지만, 세부 사항 보존에는 충분하지 않습니다. 이 맥락에서, DDIM Inversion은 이미지 콘텐츠에 대한 정보를 포함하는 초기 노이즈를 제공함으로써 중요한 역할을 합니다. 이 단계를 생략하면 추론은 이미지 콘텐츠가 없는 무작위 노이즈에 의존하게 되어 diffusion 과정에서 덜 통제 가능한 결과를 초래합니다. 그러나 inversion 과정은 불안정하고 시간 소모적입니다. 이를 완화하기 위해, 우리는 공간적 세부 사항 보존을 효과적으로 추출 및 강화하기 위해 ==SCB를 도입==합니다. 우리는 사전 학습된 VAE 인코더 \[11\] $Enc_V(\cdot)$를 사용하여 입력 이미지 $I$의 전체 콘텐츠를 캡처합니다. 이는 $F$로 표시됩니다. 이 포괄적인 이미지 세부 사항은 $F^p$로부터의 의미적 가이드와 결합하여, 우리의 SCB에 의해 처리되어 공간적 특징 $F^s$를 증류합니다:


$$F^s = SCB(F, F^t, F^p) = Att(Res(F, F^t), F^t, F^p),$$


여기서 $F^t$는 diffusion 과정에서 다양한 시간 단계를 나타내기 위해 사용되는 시간 임베딩입니다. $Res$와 $Att$ 블록은 diffusion model \[54\]의 표준 ResNet 및 Cross-Attention transformer 블록을 나타냅니다. $Res$로부터의 출력은 Query 특징으로 사용되고 $F^p$는 cross-attention layer에서 Key 및 Value 특징으로 작용합니다.

우리는 그런 다음 task-specific visual guidance prior $F^p$를 diffusion model의 cross-attention layers에 도입하여, 모델의 생성 과정을 low-level vision task의 특정 요구 사항으로 안내하는 데 사용합니다. 이를 따라, 우리는 증류된 spatial prior $F^s$를 residual로서 디코더의 마지막 단계에 직접 통합합니다. 이러한 배치는 Table 4에서 우리의 실험적 관찰에 기반한 것으로, stable diffusion \[54\]에서 공간적 세부 사항의 충실도가 얕은 층에서 깊은 층으로 갈수록 감소하는 경향이 있음을 나타냅니다. 특정 단계에서 $F^s$를 추가함으로써, 우리는 이러한 경향을 효과적으로 반전시켜 세밀한 공간적 세부 사항의 보존을 강화합니다.

Task-Plugin 모듈을 학습시키기 위해, 우리는 diffusion denoising 훈련 과정에 task-specific priors를 도입하는 \[54\]에 정의된 denoising 손실을 채택합니다:


$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, t, F^p, F^s, \epsilon \sim \mathcal{N}(0,1)} \left[ \left\| \epsilon - \epsilon_\theta (\mathbf{z}_t, t, F^p, F^s) \right\|_2^2 \right],$$


여기서 $\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t$는 시간 $t$에서 잠재 공간 이미지의 노이즈 버전을 나타내고, $\mathbf{z}_0$는 진짜 이미지 $\hat{I}$의 잠재 공간 표현으로 $Enc_V(\hat{I})$에서 얻어집니다. 이 손실 함수는 Task-Plugin이 diffusion 과정에서 가이드를 제공하는 task-specific priors를 통합하도록 효과적으로 학습되었음을 보장합니다.

## 3.4. Plugin-Selector

우리는 사용자가 텍스트 입력을 사용하여 원하는 Task-Plugin을 선택할 수 있도록 하는 Plugin-Selector를 제안합니다. 입력 이미지 $I$와 텍스트 프롬프트 $T$에 대해, 우리는 Task-Plugins의 집합을 $\mathcal{P} = \{ \mathcal{P}_1, \mathcal{P}_2, \cdots, \mathcal{P}_m \}$으로 정의하며, 각 $\mathcal{P}_i$는 특정 vision task에 해당하여, $I$를 task-specific priors ($F^p_i, F^s_i$)로 변환합니다. 그런 다음, 각 Task-Plugin의 visual guidance $F^p_i$는 공유된 visual projection head $VP(\cdot)$를 통해 새 텍스트-시각적 정렬된 multi-modality latent space로 캐스팅되며, 이는 $\mathcal{V} = \{ v_1, v_2, \cdots, v_m \}$으로 나타냅니다. 동시에, $T$는 텍스트 임베딩 $Enc_T(\cdot)$ \[49\]에 인코딩되고, 그런 다음 텍스트 projection head $TP(\cdot)$를 사용하여 $q$에 투영되어 텍스트 및 시각적 임베딩을 정렬합니다. 이 과정은 다음과 같이 공식화됩니다:


$$v_i = VP(F^p_i); \quad q = TP(Enc_T(T)).$$


우리는 그런 다음 텍스트 임베딩 $q$와 각 시각적 임베딩 $v_i \in \mathcal{V}$를 코사인 유사도 함수를 사용하여 비교하여, 유사도 점수 집합 $\mathcal{S} = \{ s_1, s_2, \cdots, s_m \}$을 도출합니다. 우리는 지정된 유사도 임계값 $\theta$를 충족하는 Task-Plugin $\mathcal{P}_{\text{selected}}$를 선택합니다:


$$\mathcal{P}_{\text{selected}} = \{ \mathcal{P}_i \mid s_i \geq \theta, \mathcal{P}_i \in \mathcal{P} \}.$$


우리는 $F^p_i$를 pseudo label로 채택하여, 이를 task-specific 텍스트와 짝지어 훈련 데이터를 구성합니다. 우리는 contrastive loss \[5, 49\]를 사용하여 비전 및 텍스트 프로젝션 헤드를 최적화하여, 다중 작업 시나리오를 처리하는 능력을 향상시킵니다. 이는 앵커 이미지와 긍정 텍스트 간의 거리를 최소화하면서 부정 텍스트로부터의 거리를 증가시키는 것을 포함합니다. 각 이미지 $I$에 대해, 작업과 관련된 긍정 텍스트(예: deraining task의 경우 "I want to remove rain")와 다른 작업에서 $N$ 개의 부정 텍스트(예: face restoration의 경우 "enhance the face")를 샘플링합니다. 긍정 예제 쌍 $(i, j)$에 대한 손실 함수는 다음과 같습니다:


$$\ell_{i,j} = -\log \frac{\exp \left( \text{sim} (v_i, q_j) / \tau \right)}{\sum_{k=1}^{N+1} \mathbb{1}_{[k_c \neq i_c]} \exp \left( \text{sim} (v_i, q_k) / \tau \right)},$$


여기서 $c$는 각 샘플의 작업 유형을 나타내며, $\mathbb{1}_{[k_c \neq i_c]} \in \{0,1\}$은 $k_c \neq i_c$일 때 1로 평가되는 지시자 함수입니다. $\tau$는 온도 매개변수를 나타냅니다.


