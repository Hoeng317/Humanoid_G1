# ACCAD-G1 정책 v4 Full 학습 진단

작성 기준 시각: 2026-08-09 (Asia/Seoul)

## 결론

v4 정책의 5,000-update full 학습은 실행·무결성 측면에서는 정상 완료되었지만,
고정 validation 5개 모션의 사전 정의 acceptance에는 실패했다. 저장된 11개
checkpoint를 전부 비교했으나 clean motion-end 성공은 모두 0/5였다. train에
포함된 Stand, B3 Walk, C3 Run 진단도 모두 중간 종료되어 단순 checkpoint 선택이나
validation OOD만으로 설명할 수 없다. 따라서 test 29개는 열지 않았으며, 다음
실험은 목표 root 속도와 미래 root 궤적을 actor 관측에 명시하는 v5로 진행한다.

## 1. Full 학습 실행 무결성

- Run: `work/training/runs/2026-08-09_04-02-27_recovery_v4_direct_full5000_seed42`
- 정책 계약: `accad_g1_tracking_recovery_v4_fixed_lr_zero_reset_survival`
- 시작/종료 UTC: `2026-08-08T19:02:27.953490+00:00` / `2026-08-08T22:45:40.379599+00:00`
- Fresh run: resume checkpoint 없음
- 환경/rollout/update: `2048 / 32 / 5000`
- 총 transition: `327,680,000`
- train motion sampling: `215/215`, 누락 0
- motion별 reset sample 최소/최대: `19,925 / 20,721`
- 총 episode reset sample: `4,367,608`
- frame-zero / continuous-random: `2,182,829 / 2,184,779`
- PPO 최종 learning rate: algorithm/optimizer 모두 `1e-4`
- 체크포인트: `model_0, 500, ..., 4500, 4999.pt` 총 11개
- actor/critic/action: `260 / 472 / 29`
- TorchScript/ONNX: 둘 다 260→29 finite inference 및 ONNX checker 통과

핵심 SHA-256:

- `run_summary.json`: `89e097df96dcafdd244060a9a38a4bc2ddae59e690772df973aeefd620d5397d`
- `runtime_contract.json`: `84c4e8144078747362d2513ba1aa0116b765593594a46b6d872bcb04ed30cbac`
- `model_4999.pt`: `8909e8ad0e0aa265735433110f45bc765048475db1399ba4e871c302ced1cf0e`
- TorchScript: `8f3ee19d4fa3a0f31d1038e74f4c381874fffb835896b3f80afec6084d12460e`
- ONNX: `7c8b7268d5bf3233c4d818aaf5802254e5956d523295bc271f3a9387c2f3fbbd`

## 2. Validation checkpoint sweep

동일 seed 42, deterministic evaluation, frame 0, state/observation noise 0,
domain randomization off, 동일 validation manifest를 사용했다. 성공 수가 모두 같으므로
completion 평균만 보면 model 3000이 가장 높지만 acceptance를 통과한 후보는 없다.

| Checkpoint | Clean success | Min completion | Mean completion | Joint MAE rad | Root error m | Body error m | Contact F1 | Raw clip |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0/5 | 0.028221 | 0.178364 | 0.076813 | 0.120149 | 0.136977 | 0.872928 | 0.000000 |
| 500 | 0/5 | 0.030675 | 0.237402 | 0.092888 | 0.130540 | 0.146096 | 0.896486 | 0.011052 |
| 1000 | 0/5 | 0.031902 | 0.241613 | 0.100932 | 0.137573 | 0.147649 | 0.890558 | 0.033876 |
| 1500 | 0/5 | 0.031902 | 0.274307 | 0.107990 | 0.142706 | 0.150247 | 0.910355 | 0.047437 |
| 2000 | 0/5 | 0.031902 | 0.269384 | 0.115287 | 0.143501 | 0.156873 | 0.895780 | 0.060545 |
| 2500 | 0/5 | 0.031902 | 0.275136 | 0.119849 | 0.146116 | 0.159675 | 0.888516 | 0.075279 |
| 3000 | 0/5 | 0.031902 | 0.285553 | 0.120334 | 0.141027 | 0.153719 | 0.920900 | 0.087725 |
| 3500 | 0/5 | 0.031902 | 0.245041 | 0.116840 | 0.146055 | 0.151667 | 0.883576 | 0.111323 |
| 4000 | 0/5 | 0.031902 | 0.259256 | 0.118081 | 0.151365 | 0.155488 | 0.897881 | 0.116864 |
| 4500 | 0/5 | 0.031902 | 0.248843 | 0.117245 | 0.140536 | 0.149722 | 0.906746 | 0.110920 |
| 4999 | 0/5 | 0.031902 | 0.253821 | 0.120663 | 0.142139 | 0.153795 | 0.893043 | 0.110203 |

model 4999의 validation report:

- 경로: `work/training/evaluations/2026-08-09_07-46-59_recovery_v4_direct_full5000_validation_model4999/suite_evaluation_summary.json`
- SHA-256: `a41c82e8272e8ce54a1bb7b701b4a82f9808c4ce938598e3e315d9be9a0cc790`
- termination 집계: fall 4, tracking divergence 3, joint limit 0, nonfinite 0
- acceptance 실패 항목: 5/5 clean success, completion, body error, raw-action clip

## 3. Train motion 진단

최종 model 4999를 train에 실제 포함된 세 가지 대표 모션에 frame 0부터 적용했다.
Direct-motion diagnostic이므로 manifest provenance check는 의도적으로 최종 acceptance에
사용하지 않고, 각 first episode의 종료 원인과 completion만 해석한다.

| Motion | Frame | 실행 step | Completion | 종료 원인 | Joint MAE rad | Root error m | Body error m | Raw clip |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| A1 Stand | 150 | 63 | 0.4200 | tracking divergence | 0.058653 | 0.191922 | 0.163933 | 0.084291 |
| B3 Walk | 381 | 176 | 0.461942 | tracking divergence | 0.118594 | 0.187941 | 0.194129 | 0.134992 |
| C3 Run | 85 | 15 | 0.176471 | fall | 0.136410 | 0.090887 | 0.168507 | 0.190805 |

Report:
`work/training/evaluations/2026-08-09_07-52-40_recovery_v4_direct_full5000_train_probe_stand_b3_c3_model4999/suite_evaluation_summary.json`

## 4. 원인 판정과 다음 변경

v4 actor는 현재 root pose/velocity의 reference-minus-actual 오차를 받지만, 목표
root 선속도·각속도 자체와 미래 root 이동/회전 궤적을 받지 않는다. 실제 상태가
reference와 순간적으로 일치할 때 정지와 고속 이동 명령이 모두 0 오차로 alias될 수
있다. train의 정지·보행·달리기까지 모두 실패했으므로 다음 변경을 채택한다.

- 실제 G1 pelvis frame의 목표 root 선속도 3-D와 각속도 3-D
- 0.04/0.08/0.16초 미래 root position error 9-D
- 같은 horizon의 미래 root orientation error 18-D
- actor/critic `260/472`에서 `293/505`로 변경
- retarget NPZ, 29-D action, reward, termination, PPO는 유지
- 실제 Isaac observation term 순서·개별 shape까지 runtime contract에 고정

v5는 먼저 fresh single-motion Stand, B3 Walk, C3 Run overfit을 각각 통과해야 한다.
그 뒤 세 모션 혼합 pilot, 마지막으로 fresh 215-motion full 학습을 진행한다.

## 5. Leakage 상태

- validation 5개는 checkpoint 선택과 구조 진단에 사용했다.
- test 29개는 이 문서 작성 시점까지 평가하지 않았다.
- test는 validation 5/5와 모든 사전 acceptance check 통과 후 한 번만 사용한다.
