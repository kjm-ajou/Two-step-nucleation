# Fe Two-Step Phase-Field 모델 — 수식·개념·설정 정리

순수 Fe의 다형 전이 **FCC(parent) → amorphous(준안정) → BCC(crystal)** 를 phase field로 푸는 모델의 이론·구현 문서. 핵심 설계는 **이론과 phase field의 분업**이다: Turnbull–Fisher two-step 핵생성 이론이 *언제·어디서·몇 개*(핵생성률·유도시간)를 정하고, multi-Allen–Cahn phase field가 *그 뒤 성장·경쟁*을 풀며, 보존 자유부피장이 *국소 고갈 피드백*을 준다.

---

## 0. 전체 구조

$$
\underbrace{J_d,\,J_{com},\,J_c,\;\theta_d,\,\theta_{com},\,\theta_c}_{\text{two-step 이론 (계산값)}}
\;\xrightarrow[\text{Simmons }P=1-e^{-J\Delta V\Delta t}]{}\;
\underbrace{\text{post-critical disk 주입}}_{\text{handoff}}
\;\xrightarrow{\text{multi-Allen–Cahn}}\;
\text{성장·경쟁}
$$

여기에 보존장 $c(\mathbf r,t)$ 가 결합되어, 구동력 $\Delta g_i(c)$ 와 핵생성률 배수 $m(c)$ 를 공간·시간에 따라 변조하고, 상이 형성되면 $c$ 를 소모한다.

---

## 1. 표준 phase field (출발점)

### 1.1 자유에너지 범함수
비보존 구조 order parameter $\eta$ (예: $\eta=0$ parent, $\eta=1$ crystal) 하나에 대해:

$$
F=\int_V\Big[\;\underbrace{h(\eta)\,\Delta g}_{\text{(1) tilt}}\;+\;\underbrace{W\,g_{dw}(\eta)}_{\text{(2) double well}}\;+\;\underbrace{\tfrac{\kappa}{2}\,|\nabla\eta|^2}_{\text{(3) gradient}}\;\Big]\,dV
$$

- **(1) tilt(구동력) 항**: 두 상의 자유에너지 차 $\Delta g$ 를 우물에 실어 어느 상이 안정한지 정한다. $h$ 는 0→1 스위치, $\Delta g$ 는 깊이 차.
- **(2) double well(장벽) 항**: 두 상을 분리하는 극소 두 개와 그 사이 장벽(높이 $W/16$)을 만든다.
- **(3) gradient(계면) 항**: 공간적으로 급변하는 곳(계면)에 에너지 비용을 매겨 계면 폭과 에너지를 유한하게 만든다.

### 1.2 운동방정식 (Allen–Cahn, 비보존)

$$
\frac{\partial\eta}{\partial t}=-L\,\frac{\delta F}{\delta\eta}
=-L\big(h'(\eta)\,\Delta g+W\,g_{dw}'(\eta)-\kappa\nabla^2\eta\big)
$$

$L$ 은 mobility(계면 운동 속도). **비보존**이란 $\int\eta\,dV$ 가 시간에 따라 보존되지 않는다는 뜻 — 상이 생겨나므로 당연하다.

### 1.3 보조 함수 — 형태는 끝점 조건으로 유도됨 (임의 아님)

**보간함수**
$$
h(\eta)=\eta^3(10-15\eta+6\eta^2),\qquad h'(\eta)=30\,\eta^2(1-\eta)^2
$$
다섯 조건 $h(0)=0,\;h(1)=1,\;h'(0)=h'(1)=0,\;h''(0)=h''(1)=0$ 을 만족하는 최소 차수(quintic). 특히 $h'(0)=h'(1)=0$ 이라 **tilt를 줘도 우물 위치($\eta=0,1$)가 안 변한다** → 계면 폭·에너지가 $\Delta g$ 에 오염되지 않음.

**double well**
$$
g_{dw}(\eta)=\eta^2(1-\eta)^2,\qquad g_{dw}'(\eta)=2\eta(1-\eta)(1-2\eta)
$$
조건 $g(0)=g(1)=0,\;g'(0)=g'(1)=0,\;g'(\tfrac12)=0$ 을 만족하는 최소 차수(quartic).

### 1.4 계면 물성과 임계반지름

$$
\delta=\sqrt{\frac{\kappa}{2W}}\quad(\text{계면 폭}),\qquad
\sigma=\frac{\sqrt{\kappa W}}{3\sqrt2}\quad(\text{계면에너지}),\qquad
R^*=\frac{\sigma}{|\Delta g|}\quad(\text{Gibbs–Thomson})
$$

평형 계면은 $\eta(x)=\tfrac12\big[1+\tanh(x/2\delta)\big]$. $R^*$ 보다 큰 씨앗은 자라고 작은 씨앗은 사라진다.

---

## 2. two-step을 담기 위한 수식 변형

표준 모델은 우물이 둘뿐이라 parent→crystal 직행만 가능하다. 중간 상(amorphous)을 담으려면 구조를 바꿔야 한다.

### 2.1 order parameter 2개
$\eta_m$ (amorphous), $\eta_c$ (BCC) 두 개를 도입한다. 상태 정의:

| 상태 | $(\eta_m,\eta_c)$ |
|---|---|
| parent (FCC) | $(0,0)$ |
| amorphous (준안정) | $(1,0)$ |
| crystal (BCC, 안정) | $(0,1)$ |
| 비물리적 (금지) | $(1,1)$ |

→ 자유에너지 지형에 **우물이 셋** 생긴다.

### 2.2 변형된 자유에너지

$$
F=\int_V\Big[\sum_{i\in\{m,c\}} h(\eta_i)\,\Delta g_i(c)
+ W\Big(\sum_i g_{dw}(\eta_i)+\underbrace{\alpha\,\eta_m^2\eta_c^2}_{\text{(★) 상호 배척}}\Big)
+ \tfrac{\kappa}{2}\sum_i|\nabla\eta_i|^2
+ \underbrace{\tfrac{\kappa_c}{2}|\nabla c|^2}_{\text{보존장 gradient}}\Big]\,dV
$$

표준 대비 추가/변경된 것:
- 상별 독립 tilt $\sum_i h(\eta_i)\Delta g_i(c)$ — 구동력이 보존장 $c$ 의 함수.
- **(★) 상호 배척 항 $\alpha\,\eta_m^2\eta_c^2$** (신규): 한 셀이 두 상으로 동시에 가득 차는 $(1,1)$ 을 에너지 봉우리로 만들어 막는다. $\eta_m+\eta_c\le1$ 을 부드럽게 강제(부등식 구속이 아니라 페널티).
- 보존장 gradient $\tfrac{\kappa_c}{2}|\nabla c|^2$ (Model C의 완전형). *현재 구현은 단순화하여 $\kappa_c=0$ 로 두고 §2.5의 reaction–diffusion 형태를 쓴다.*

### 2.3 변형된 운동방정식 — multi-Allen–Cahn

$$
\frac{\partial\eta_m}{\partial t}=-L_m\frac{\delta F}{\delta\eta_m},\qquad
\frac{\partial\eta_c}{\partial t}=-L_c\frac{\delta F}{\delta\eta_c}
$$

비선형(국소) 항을 전개하면 (배척 항이 교차로 들어감):

$$
N_m \equiv \Delta g_m\,h'(\eta_m)+W\big(g_{dw}'(\eta_m)+2\alpha\,\eta_m\eta_c^2\big)
$$
$$
N_c \equiv \Delta g_c\,h'(\eta_c)+W\big(g_{dw}'(\eta_c)+2\alpha\,\eta_c\eta_m^2\big)
$$

$2\alpha\eta_m\eta_c^2$, $2\alpha\eta_c\eta_m^2$ 가 두 order parameter를 서로 밀어내게 한다.

### 2.4 3채널 핵생성 (경쟁)

표준의 단일 noise 시딩 대신, master equation의 세 rate를 각자 자기 셀 조건·유도시간으로 평가한다. 셀당·step당 변환 확률(Simmons):

$$
P=1-\exp\!\big[-J\cdot m(c)\cdot\Delta V\cdot\Delta t_{\rm phys}\big],\qquad
\Delta V=(\Delta x)^2\times a_0
$$

세 채널:

| 채널 | rate | 유도시간 게이트 | 셀 조건 (위상 게이트) |
|---|---|---|---|
| parent → amorphous | $J_d$ | $t>\theta_d$ | $\eta_m+\eta_c<0.5$ (parent) |
| parent → crystal (직접) | $J_c$ | $t>\theta_c$ | $\eta_m+\eta_c<0.5$ (parent) |
| amorphous → crystal | $J_{com}$ | $t>\theta_{com}$ | $\eta_m>0.5,\ \eta_c<0.5$ (amorphous) |

crystal-in-amorphous 시딩 직후 amorphous를 소모: $\eta_m\leftarrow\min(\eta_m,\,1-\eta_c)$.

핵심: 세 경로를 모두 *열어두고*, 어느 경로가 쓰이는지는 차단이 아니라 rate가 통계적으로 정한다. $J_d/J_c\approx2.2\times10^{19}$ 라 직접경로는 통계적으로 침묵하고, $J_d/J_{com}\approx8200$ 이라 amorphous 내부에서 crystal이 등장한다 → "amorphous 우세 → 그 안에서 crystal"이 *창발*한다.

**핵생성률 공간 변조** — CNT 장벽의 함수:
$$
W^*_m=\frac{4\gamma_{mo}^3}{27\,(c-s_{cm})^2},\qquad
W^*_c=\frac{4\gamma_{co}^3}{27\,c^2},\qquad
m(c)=\frac{\exp\!\big[-(W^*(c)-W^*(c_0))\big]}{\big\langle\exp[-(W^*(c)-W^*(c_0))]\big\rangle}
$$
공간평균을 1로 정규화하므로 **공간평균 rate는 stationary 값 그대로**, 고-$c$ 셀이 우선 핵생성한다. $c$ 가 문턱 아래로 완전 고갈되면 $m\to0$ (핵생성 정지).

### 2.5 보존 자유부피장 (depletion feedback) — 이 모델의 핵심 추가

고정 supersaturation 장 대신, 보존되는 국소 구동 reservoir $c(\mathbf r,t)$ 를 시간 발전시킨다.

**원리적 형태 (Model C, Cahn–Hilliard):**
$$
\frac{\partial c}{\partial t}=\nabla\!\cdot\!\big(M\nabla\mu\big),\qquad
\mu=\frac{\delta F}{\delta c}=\sum_i h(\eta_i)\frac{\partial\Delta g_i}{\partial c}-\kappa_c\nabla^2 c
$$

**구현 형태 (최소, reaction–diffusion, $\kappa_c=0$):**
$$
\frac{\partial c}{\partial t}=\underbrace{D_c\nabla^2 c}_{\text{확산(보충)}}-\underbrace{\sum_i\lambda_i\frac{\partial\eta_i}{\partial t}}_{\text{소모(sink)}}
\qquad\Longrightarrow\qquad
\int_V\Big(c+\sum_i\lambda_i\eta_i\Big)dV \ \text{보존}
$$

- 구동력은 매 step **현재 $c$** 로 재계산: $\Delta g_m=-\max(c-s_{cm},0)\cdot\text{scale}$, $\Delta g_c=-\max(c,0)\cdot\text{scale}$.
- $c$ 의 정체 = **국소 자유부피/구동 reservoir**. 수송 속도(확산도)는 amorphous 확산 $D_g$ 스케일이라, 작은 nm/µs 영역에서 고갈 halo가 국소에 남는다.
- 결과: 먼저 생긴 핵 주변은 $c$ 가 줄어 이웃의 핵생성·성장이 억제됨 = **soft impingement**. 핵이 서로 밀어내며 유한한 grain 간격을 갖는다.

---

## 3. 핵심 개념

### 3.1 Two-step nucleation & Ostwald step rule
crystal이 parent에서 직접 나오지 않고, 먼저 준안정상(amorphous)을 거쳐 그 안에서 나타나는 두 단계 경로. Ostwald step rule("가장 안정한 상이 아니라 자유에너지가 가장 가까운 상이 먼저 나타난다")의 동역학적 실현. 여기서는 $J_d\gg J_c$ 가 그 이유를 정량적으로 제공한다.

### 3.2 우물 지형 (triple well)
표준은 우물 둘(parent, crystal), two-step은 셋(parent / amorphous(준안정) / crystal(안정)). amorphous는 parent보다 깊고 crystal보다 얕은 **중간 우물**. 깊이 순서($0>\Delta g_m>\Delta g_c$)는 자유에너지가 주지만, "amorphous를 경유한다"는 *밟는 순서*는 장벽 설계가 아니라 3채널 핵생성률이 정한다(hybrid의 본질, §3.7).

### 3.3 보존 vs 비보존 (Model A/B/C), 무한/유한 공급
- **비보존(Allen–Cahn, Model A)**: $\dot\eta=-L\,\delta F/\delta\eta$. $\int\eta$ 보존 안 됨. order parameter $\eta_m,\eta_c$ 가 이 종류.
- **보존(Cahn–Hilliard, Model B)**: $\dot c=\nabla\!\cdot(M\nabla\mu)$. divergence 형태라 $\int c$ 보존. 보존장 $c$ 가 이 종류.
- **Model C**: 보존 + 비보존 결합. 이 모델의 전체 구조.

보존/비보존은 *어떤 장이 보존되느냐*의 성질이지 parent phase의 보존 얘기가 아니다(parent의 양은 변태로 줄어들 뿐). "$c$ 가 보존된다 = $c$ 공급이 유한하다"는 동전의 양면 — 고정 랜덤장(`conserved=False`)은 사실상 무한 공급, 보존장은 유한 공급 + 국소 고갈.

### 3.4 Soft impingement & 고갈 halo
보존장이 유한하므로 한 곳에서 많이 쓰면 주변이 부족해진다. 핵 주변에 $c$ 가 파인 halo가 생겨 이웃 성장·핵생성을 억제한다. 확산이 느릴수록(용질·자유부피) halo가 좁고 뚜렷하고, 빠를수록(열) 넓게 번져 전역화된다.

### 3.5 Supersaturation 장 & CNT 장벽 기반 rate 변조
구동력 $s=\Delta\mu/k_BT$ 의 공간 분포. 고-$s$ 영역은 구동력 $\Delta g$ 도 크고 CNT 장벽 $W^*\propto1/s^2$ 도 낮아 핵생성률 배수 $m(s)$ 도 크다 → 우선 핵생성. 공간평균=1 정규화로 stationary 절대값은 보존.

### 3.6 Simmons handoff (sub-nm 임계핵 → post-critical disk)
임계핵 $r^*$ 가 sub-nm(아래 §5.3)이라 diffuse interface(계면 폭 $\sim$ 격자 2칸)로 직접 분해 불가. 따라서 핵생성을 Simmons 확률로 평가하고, 발화 시 **이미 임계를 넘긴 작은 disk**를 주입한다(직접 임계핵을 풀지 않음).

### 3.7 경로 강제: 자유에너지 vs 시딩 마스크 (hybrid의 본질)
정통 접근은 경로를 자유에너지 지형에 새기는 것(직행 장벽 ↑, 경유 장벽 ↓)이다. 이 모델은 대신 핵생성 물리를 이론에서 이미 풀었으므로, 그 결과($J_d\gg J_c$, 경로 위상)를 **시딩 마스크로 직접 부과**하는 hybrid를 쓴다. 장점: sub-nm 임계핵을 격자에서 분해할 필요 없이 정량 rate를 그대로 사용. 한계: 경로가 자유에너지에서 창발하는 게 아니라 부과된 것 — "왜 그 경로인가"의 답은 phase field가 아니라 two-step 이론이 갖고 있다.

---

## 4. 수치 기법

### 4.1 Spectral semi-implicit
선형 항만 Fourier 공간에서 implicit 처리. **비보존(Allen–Cahn)** 과 **보존(확산)** 의 implicit 인수 구조가 다르다:

$$
\text{Allen–Cahn:}\quad \hat\eta^{n+1}=\frac{\hat\eta^{n}-\Delta t\,L\,\widehat{N}}{1+\Delta t\,L\,\kappa\,k^2}
\qquad\qquad
\text{확산(보존):}\quad \hat c^{n+1}=\frac{\hat c^{n}-\widehat{\text{sink}}}{1+\Delta t_g\,D_c\,k^2}
$$

(완전 Cahn–Hilliard라면 분모가 $1+\Delta t\,M\kappa_c\,k^4$ 로 biharmonic.) 두 인수 모두 $k^2$ 구조라 같은 솔버 골격에 들어간다. sink $=\lambda_m\Delta\eta_m+\lambda_c\Delta\eta_c$.

### 4.2 무차원화
- $W=1$ (에너지 단위 정규화).
- $\kappa=2W\ell^2$, $\ell=2\Delta x$ → 계면 폭을 격자 2칸으로 고정(해상도 결정). $\delta,\sigma$ 가 자동으로 따라옴.
- $L_m,L_c,\alpha$ → phenomenological 튜닝(비율 $L_m/L_c$ 가 상대 성장 경쟁).
- 보존장 확산도 무차원화:
$$
D_c=\frac{D_g\,\Delta t_{\rm phys}}{dt_g}\times10^{18}\approx 4\qquad(D_g\approx1.6\times10^{-13}\,\mathrm{m^2/s})
$$
확산길이 $\sqrt{D_g\,t_{\rm end}}\approx 18$ nm — 50 nm 박스 안이라 국소 halo가 보인다.
- $\lambda$ 는 단위 변태당 $s$ 감소량이라 $\mathcal O(1)$ (한 셀 완전 변태 시 $s$ 가 문턱 $s_{cm}$ 아래로 떨어질 정도).

---

## 5. 파라미터·설정 (실제 값)

### 5.1 격자·시간
| 기호 | 값 | 의미 |
|---|---|---|
| $N$ | 170 | 격자 한 변 |
| $L_{\rm box}$ | 50 nm | 박스 크기 |
| $\Delta x$ | $50/170\approx0.294$ nm | 격자 간격 |
| $t_{\rm end}$ | $2\times10^{-3}$ s | 총 물리 시간 |
| $n_{\rm steps}$ | 4000 | step 수 |
| $\Delta t_{\rm phys}$ | $5\times10^{-7}$ s | 물리 step |
| $dt_g$ | 0.02 | 무차원 성장 step |
| seed | 7 | 난수 시드 |
| $\xi$ (corr_len) | 6 nm | 랜덤장 상관길이 |
| $s_{\rm std}$ | 0.45 | 랜덤장 표준편차 |
| seed_disk | 1.0 nm | 주입 disk 반지름 |

### 5.2 phase-field (우물·계면·mobility)
| 기호 | 값 | 의미 |
|---|---|---|
| $W$ | 1.0 | double well 높이(정규화) |
| $\kappa$ | $2W\ell^2,\ \ell=2\Delta x$ | gradient 계수(해상도로 고정) |
| $L_m$ | 0.6 | amorphous mobility |
| $L_c$ | 1.0 | crystal mobility |
| $\alpha$ | 2.0 | 상호 배척 강도 |
| drive_scale | 0.02 | $\Delta g$ 스케일(phenomenological) |

### 5.3 Fe 열역학 스케일 (160 K)
| 기호 | 값 | 의미 |
|---|---|---|
| $T$ | 160 K | 온도 |
| $s_{co}$ | 6.559 | 평균 supersaturation (parent→crystal 기준) |
| $s_{cm}$ | 5.743 | amorphous 구동력 문턱 |
| $s_{mo}=s_{co}-s_{cm}$ | 0.816 | amorphous 구동 여유 |
| $\gamma_{mo}$ | 2.826 | 무차원 계면 파라미터(amorphous/parent) |
| $\gamma_{co}$ | 27.944 | 무차원 계면 파라미터(BCC/parent) |
| $V_O$ | $11.55\times10^{-30}$ m³ | 원자 부피 |
| $a_0=V_O^{1/3}$ | 0.226 nm | 원자 간격 |
| $\sigma_{mo}$ | ≈ 25.3 mJ/m² | amorphous/parent 계면에너지 |
| $\sigma_{co}$ | ≈ 249.8 mJ/m² | BCC/parent 계면에너지 |
| $r^*_{\rm amorphous}$ | ≈ 0.324 nm | amorphous 임계반지름 |
| $r^*_{\rm BCC}$ | ≈ 0.398 nm | BCC 임계반지름 |

$\sigma_{mo}\ll\sigma_{co}$ → amorphous 장벽이 훨씬 낮음 → Ostwald step 예상. $r^*$ 가 모두 sub-nm → Simmons handoff 필요.

### 5.4 핵생성 rate·유도시간 (Turnbull–Fisher 계산값)
| 기호 | 값 | 의미 |
|---|---|---|
| $J_d$ | $4.31\times10^{35}$ m⁻³s⁻¹ | parent → amorphous |
| $J_{com}$ | $5.27\times10^{31}$ m⁻³s⁻¹ | amorphous → crystal |
| $J_c$ | $1.97\times10^{16}$ m⁻³s⁻¹ | parent → crystal (직접) |
| $\theta_d$ | $1.71\times10^{-5}$ s | amorphous 유도시간 |
| $\theta_{com}$ | $1.39\times10^{-4}$ s | crystal-in-amorphous 유도시간 |
| $\theta_c$ | (placeholder $=\theta_{com}$) | 직접경로 유도시간 — two-step `induction_df`의 `_theta_c`로 교체 권장 |

비율: $J_d/J_{com}\approx8200$, $J_d/J_c\approx2.2\times10^{19}$.

### 5.5 보존 자유부피장
| 기호 | 값 | 의미 |
|---|---|---|
| $D_c$ | 4.0 | 무차원 확산도($D_g$ 에서 환산, 확산길이 ≈18 nm) |
| $\lambda_m$ | 1.0 | amorphous 형성당 $c$ 소모 |
| $\lambda_c$ | 1.3 | crystal 형성당 $c$ 소모(더 안정 → 큼) |
| conserved | True / False | 보존장 발전 on/off (off = 고정장 대조군) |

---

## 6. 검증 (노트북 내장)

1. **평형 계면 폭** — 구동력 0으로 슬랩 이완 시 27–73% 폭이 이론값 $2\delta$ 와 일치하는지.
2. **Gibbs–Thomson 임계반지름** — 고정 $\Delta g$ 에서 $R^*=\sigma/|\Delta g|$ 위는 자라고 아래는 사라지는지.

이 두 검증이 "골라낸 무차원 값들이 의도한 $\delta,\sigma,R^*$ 를 실제로 재현하는지"를 보증한다(정량 핵생성 물리는 별도로 two-step 이론이 공급).

---

## 7. 한계·주의

- **경로는 부과된 것**: amorphous 경유가 자유에너지에서 창발하지 않고 시딩 마스크로 강제됨(§3.7). 정통화하려면 자유에너지 지형에 비대칭 장벽을 새겨야 함.
- **$c$ 는 conserved 조성이 아님**: 순수 Fe라 실제 보존 조성은 없음. $c$ 는 "국소 구동 reservoir(자유부피 해석)"로, $D_c$ 를 자유부피/amorphous 확산 스케일로 잡은 모델링 선택.
- **잠열 해석은 전역적**: 잠열로 보면 열확산이 너무 빨라($\sqrt{\alpha_{th}t_{\rm end}}\sim10^2\,\mu$m ≫ 박스) 국소 halo가 안 생기고 전역 recalescence가 됨 → field가 아니라 mean-field $T(t)$ 로 다뤄야 맞음. 그래서 국소성을 보려면 자유부피 해석이 적합.
- **phenomenological 스케일**: drive_scale, $D_c$, $\lambda$ 는 무차원 튜닝값. 완전 물리 단위 calibration은 별도 작업.
- **직접 crystal은 통계적으로 0**: $J_c$ 가 워낙 작아 이 박스/시간에선 직접 핵 0개가 정상(경로는 열려 있되 침묵).
- **최종 상태는 BCC**: 안정상이라 transient(amorphous 정점 → BCC 전환) 뒤 결국 전 영역 BCC가 됨(Ostwald 그림과 일치).
