# HRI Dataset Inventory

이 문서는 `v3_chan/trajectories` 아래에 있는 HRI trajectory HDF5 파일들의 대략적인 상태를 정리한 것이다.

## 기준

- `1 episode`는 큐브 3개를 모두 쌓는 한 cycle이다.
- 정상 HDF5는 `/episodes/episode_xxxxxx` 그룹을 가진다.
- `obs_policy`는 기존 robot-only task policy와 호환되는 84차원 관측이다.
- `hri_obs_policy`는 HRI/safety residual policy 학습용 74차원 관측이다.
- `human_robot_collision`은 물리 contact force가 아니라 거리/프록시 기반 safety violation event로 해석한다.
- 실제 task 성공 여부는 컨트롤러의 `success` 속성만 사용하지 않고, 각 pick에서 큐브가 5 cm 이상 상승했는지와 최종 스택 위치를 함께 검사한다.
- 이후 생성되는 `hri_obs_v2_verified_place_retry` 파일은 grasp 관측, 5 cm 이상 lift, 목표 XY/Z 오차, 최종 속도를 runtime에서 검증한다. 실패한 attempt는 같은 episode 안에서 재시도되므로 episode 길이가 2751 step보다 길어질 수 있다.

## 파일 요약

| 파일 | 크기 | 상태 | episodes | 용도 |
|---|---:|---|---:|---|
| `v3_chan/trajectories/hri_vr_sphere_obs.hdf5` | 2.5 MB | 정상 | 1 | 초기 테스트 데이터. 접촉/햅틱 비율이 상대적으로 높아 stress 성격이 있음 |
| `v3_chan/trajectories/hri_vr_sphere_obs_20260711_163249_371518687_154755.hdf5` | 7.2 MB | 정상 | 3 | 현재 기준 clean HRI 수집 데이터. 초기 학습에 사용 가능 |
| `v3_chan/trajectories/hri_vr_sphere_obs_20260711_164903_675435892_154755.hdf5` | 7.2 MB | 정상 | 3 | clean HRI 수집 데이터. near/contact가 좌우에 비교적 고르게 포함됨 |
| `v3_chan/trajectories/hri_vr_sphere_obs_20260711_170231_000168904_154755.hdf5` | 7.2 MB | 부분 사용 | 3 | `episode_000001`에 실제 pick 실패가 있어 task 학습에서 제외. 나머지 2 episodes는 사용 가능 |
| `v3_chan/trajectories/hri_vr_sphere_obs_20260711_171532_116442838_154755.hdf5` | 7.2 MB | 정상/contact-rich | 3 | 실제 pick 9회 모두 성공. 일부 깊은 프록시 겹침이 포함됨 |
| `v3_chan/trajectories/failed_hri_vr_sphere_obs_20260711_160035_video_test.hdf5` | 96 B | 실패 | 0 | 내부 video recorder 테스트 중 깨진 파일. 학습 제외 |
| `v3_chan/trajectories/failed_hri_vr_sphere_obs_20260711_160035_recorder_disabled.hdf5` | 96 B | 실패 | 0 | 깨진 HDF5를 다시 열면서 recorder가 꺼진 run. 학습 제외 |
| `v3_chan/trajectories/hri_vr_sphere_obs_20260711_162231_309259908_329207.hdf5` | 96 B | 실패 | 0 | 1 episode 완료 전 종료된 테스트. 학습 제외 |

## 정상 파일 상세

### `hri_vr_sphere_obs.hdf5`

| 항목 | 값 |
|---|---:|
| schema_version | `hri_obs_v1_cognitive_safety` |
| observation_dim | 84 |
| hri_observation_dim | 74 |
| episodes | 1 |

| episode | success | length | near_human | human_robot_collision | haptic_pulse_left | haptic_pulse_right | min distance | mean distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `episode_000000` | True | 2751 | 409 | 66 | 29 | 37 | 0.0104 m | 0.2377 m |

메모:

- 첫 번째 수집 데이터이다.
- 접촉 및 햅틱 이벤트가 비교적 많다.
- 손을 강하게 넣거나 오래 겹친 구간이 포함되어 있을 수 있으므로, 초기 clean safety residual 학습에서는 비중을 낮추거나 stress/test split으로 분리하는 것을 권장한다.

### `hri_vr_sphere_obs_20260711_163249_371518687_154755.hdf5`

| 항목 | 값 |
|---|---:|
| schema_version | `hri_obs_v1_cognitive_safety` |
| observation_dim | 84 |
| hri_observation_dim | 74 |
| episodes | 3 |
| total steps | 8253 |
| head valid rate | 100% |
| left hand valid rate | 100% |
| right hand valid rate | 100% |

| episode | success | length | near_human | human_robot_collision | haptic_pulse_left | haptic_pulse_right | min distance | mean distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `episode_000000` | True | 2751 | 140 | 9 | 8 | 1 | 0.0405 m | 0.2644 m |
| `episode_000001` | True | 2751 | 260 | 7 | 5 | 2 | 0.0387 m | 0.3200 m |
| `episode_000002` | True | 2751 | 107 | 15 | 0 | 15 | 0.0220 m | 0.3140 m |

합산:

| 항목 | 값 |
|---|---:|
| total steps | 8253 |
| near_human | 507 |
| human_robot_collision | 31 |
| haptic_pulse_left | 13 |
| haptic_pulse_right | 18 |
| near_human rate | 6.14% |
| collision rate | 0.38% |

메모:

- 3 episode 모두 성공적으로 저장되었다.
- `obs_policy` shape는 각 episode `(2751, 84)`이다.
- `hri_obs_policy` shape는 각 episode `(2751, 74)`이다.
- 초기 safety residual policy 학습에 사용할 수 있는 현재 기준 clean dataset이다.
- collision 비율이 낮은 편이므로, 다음 수집에서는 near 상태는 유지하면서 light contact/haptic 이벤트를 episode당 조금 더 포함해도 좋다.

### `hri_vr_sphere_obs_20260711_164903_675435892_154755.hdf5`

| 항목 | 값 |
|---|---:|
| schema_version | `hri_obs_v1_cognitive_safety` |
| observation_dim | 84 |
| hri_observation_dim | 74 |
| episodes | 3 |
| total steps | 8253 |
| head valid rate | 100% |
| left hand valid rate | 100% |
| right hand valid rate | 100% |

| episode | success | length | near_human | human_robot_collision | haptic_pulse_left | haptic_pulse_right | min distance | mean distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `episode_000000` | True | 2751 | 299 | 7 | 0 | 7 | 0.0424 m | 0.2554 m |
| `episode_000001` | True | 2751 | 222 | 12 | 2 | 10 | 0.0440 m | 0.2506 m |
| `episode_000002` | True | 2751 | 233 | 16 | 15 | 1 | 0.0315 m | 0.2611 m |

합산:

| 항목 | 값 |
|---|---:|
| total steps | 8253 |
| near_human | 754 |
| human_robot_collision | 35 |
| haptic_pulse_left | 17 |
| haptic_pulse_right | 18 |
| near_human rate | 9.14% |
| collision rate | 0.42% |

메모:

- 3 episode 모두 성공적으로 저장되었다.
- 좌우 햅틱 합계가 각각 17회와 18회로 균형이 좋다.
- clean safety residual 학습 데이터에 포함할 수 있다.

### `hri_vr_sphere_obs_20260711_170231_000168904_154755.hdf5`

| 항목 | 값 |
|---|---:|
| schema_version | `hri_obs_v1_cognitive_safety` |
| observation_dim | 84 |
| hri_observation_dim | 74 |
| episodes | 3 |
| total steps | 8253 |
| head valid rate | 100% |
| left hand valid rate | 100% |
| right hand valid rate | 100% |

| episode | success | length | near_human | human_robot_collision | haptic_pulse_left | haptic_pulse_right | min distance | mean distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `episode_000000` | True | 2751 | 414 | 18 | 8 | 10 | 0.0301 m | 0.2348 m |
| `episode_000001` | True | 2751 | 250 | 31 | 9 | 22 | 0.0333 m | 0.2465 m |
| `episode_000002` | True | 2751 | 347 | 23 | 0 | 23 | 0.0307 m | 0.2725 m |

합산:

| 항목 | 값 |
|---|---:|
| total steps | 8253 |
| near_human | 1011 |
| human_robot_collision | 72 |
| haptic_pulse_left | 17 |
| haptic_pulse_right | 55 |
| near_human rate | 12.25% |
| collision rate | 0.87% |

메모:

- 3 episode 모두 성공적으로 저장되었고, 각 episode 종료 직후 HDF5 flush가 수행되었다.
- 최대 3 episode 종료 조건에 따라 수집 프로그램도 정상 종료되었다.
- 앞선 두 clean 세션보다 near/contact 비율이 높아 safety event 학습에 유용하다.
- 오른손 햅틱이 55회로 왼손 17회보다 많다. 이후 세션에서는 왼손 접근과 가벼운 접촉도 의도적으로 포함해 전체 데이터의 좌우 균형을 맞춘다.
- `episode_000001`의 두 번째 큐브는 2.1 mm만 상승하고 수평으로 약 7.3 cm 밀려 실제 집기에 실패했다.
- 이 실패 때문에 세 번째 큐브는 집어서 옮겼지만 3단 스택은 완성되지 않았다.
- `episode_000001`은 task BC/expert 학습과 task 성공률 평가에서 제외하거나 실패로 표시한다. HRI safety-only 학습에는 별도 품질 태그와 함께 제한적으로 사용할 수 있다.

### `hri_vr_sphere_obs_20260711_171532_116442838_154755.hdf5`

| 항목 | 값 |
|---|---:|
| schema_version | `hri_obs_v1_cognitive_safety` |
| observation_dim | 84 |
| hri_observation_dim | 74 |
| episodes | 3 |
| total steps | 8253 |
| actual successful picks | 9 / 9 |
| actual successful episodes | 3 / 3 |
| head valid rate | 100% |
| left hand valid rate | 100% |
| right hand valid rate | 100% |

| episode | actual task success | length | near_human | human_robot_collision | haptic_pulse_left | haptic_pulse_right | min distance | mean distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `episode_000000` | True | 2751 | 398 | 43 | 30 | 13 | 0.0213 m | 0.2488 m |
| `episode_000001` | True | 2751 | 222 | 36 | 17 | 19 | 0.0023 m | 0.2510 m |
| `episode_000002` | True | 2751 | 443 | 34 | 22 | 12 | 0.0223 m | 0.2269 m |

합산:

| 항목 | 값 |
|---|---:|
| total steps | 8253 |
| near_human | 1063 |
| human_robot_collision | 113 |
| haptic_pulse_left | 69 |
| haptic_pulse_right | 44 |
| near_human rate | 12.88% |
| collision rate | 1.37% |

메모:

- 3 episode와 9회의 실제 pick이 모두 성공했다.
- 모든 큐브의 최종 스택 중심 오차는 약 6 mm 이내였다.
- 3 episode 모두 종료 직후 저장되었고, 최대 episode 조건에 따라 프로그램이 정상 종료되었다.
- safety event 비율이 앞선 세션보다 높아 safety residual 학습에 유용하다.
- 좌우 햅틱은 왼손 69회, 오른손 44회로 최신 세션 안에서는 왼손 비중이 더 높다.
- 최대 gripper penetration은 왼손 약 3.8 cm, 오른손 약 5.2 cm이다. task 데이터는 정상이지만 safety 관점에서는 contact-rich/stress 성격이 있으므로 낮은 접촉 비율의 세션과 함께 사용한다.

## Clean 데이터 누적 현황

실제 pick 및 적층 결과까지 검증한 episode만 기본 clean training pool로 사용한다.

```text
전체 사용:
  hri_vr_sphere_obs_20260711_163249_371518687_154755.hdf5
  hri_vr_sphere_obs_20260711_164903_675435892_154755.hdf5
  hri_vr_sphere_obs_20260711_171532_116442838_154755.hdf5

episode 선택 사용:
  hri_vr_sphere_obs_20260711_170231_000168904_154755.hdf5
    include: episode_000000, episode_000002
    exclude from task learning: episode_000001
```

| 항목 | 누적 값 |
|---|---:|
| sessions | 4 |
| clean episodes | 11 |
| total steps | 30261 |
| near_human | 3085 |
| human_robot_collision | 220 |
| haptic_pulse_left | 107 |
| haptic_pulse_right | 113 |
| near_human rate | 10.19% |
| collision rate | 0.73% |

## 실패 파일

아래 파일들은 HDF5 superblock이 깨져 열리지 않는다.

```text
v3_chan/trajectories/failed_hri_vr_sphere_obs_20260711_160035_video_test.hdf5
v3_chan/trajectories/failed_hri_vr_sphere_obs_20260711_160035_recorder_disabled.hdf5
v3_chan/trajectories/hri_vr_sphere_obs_20260711_162231_309259908_329207.hdf5
```

공통 오류:

```text
Unable to synchronously open file
truncated file: eof = 96, stored_eof = 2048
```

해석:

- 내부 video recorder 테스트 또는 1 episode 완료 전 종료로 인해 실제 episode dataset이 저장되지 않은 파일이다.
- 학습, 평가, 통계 산출에서 제외한다.

## 현재 추천 사용법

초기 학습용:

```text
use:
  v3_chan/trajectories/hri_vr_sphere_obs_20260711_163249_371518687_154755.hdf5
  v3_chan/trajectories/hri_vr_sphere_obs_20260711_164903_675435892_154755.hdf5
  v3_chan/trajectories/hri_vr_sphere_obs_20260711_171532_116442838_154755.hdf5

use with episode filter:
  v3_chan/trajectories/hri_vr_sphere_obs_20260711_170231_000168904_154755.hdf5
  include: episode_000000, episode_000002
  exclude from task learning: episode_000001

optional stress/test:
  v3_chan/trajectories/hri_vr_sphere_obs.hdf5

exclude:
  failed_*.hdf5
  96 B truncated files
```

추가 수집 목표:

```text
normal: 50~60%
near_human: 25~35%
light contact/haptic: 5~15%
deep overlap: 최소화
```

수집 후 확인할 로그:

```text
[HRI] saved episode /episodes/episode_000000 file=... size=...MB
[HRI] saved episode /episodes/episode_000001 file=... size=...MB
[HRI] saved episode /episodes/episode_000002 file=... size=...MB
[HRI] reached max episodes (3/3); stopping collection.
```
