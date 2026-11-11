# Product Requirements Document (PRD)

## 안전한 인간-로봇 협업을 위한 Safety-enhanced Robotic Transformer (SeRT) 학습 플랫폼

**Version:** 1.0  
**Date:** 2026년 3월 13일  
**Author:** 임찬주  
**Stakeholders:** 연구팀, DAU 연구처, 산업 파트너

---

## 1. Executive Summary

### Product Vision
안전한 인간-로봇 협업 환경에서 사고를 능동적으로 예방하고 회피할 수 있는 AI 플랫폼을 개발하여, 협동 로봇 시장의 안전성과 효율성을 혁신한다.

### Key Objectives
- 인간-로봇 협업 시 충돌을 능동적으로 예방하는 Safety-enhanced Robotic Transformer 개발
- EEG 기반 암묵적 Human Feedback을 통합한 학습 플랫폼 구축
- Calvin Benchmark Test에서 80% 이상의 성공률 달성 (SOTA 78.4% 상회)

### Success Metrics
- Calvin Benchmark Test 성공률: 80%
- 충돌회피 성공률: 85%
- EEG 피드백 정확도: 90%
- SCIE 저널 게재 (JCR 25% 이내)

---

## 2. Product Overview

### Why This Product Exists (존재 이유)
협동 로봇 시장의 급성장(2026년 10조원 규모)에도 불구하고, 인간-로봇 협업 시 사고 예방이 불충분하여 산업 현장에서의 안전 문제가 심각합니다. 기존 기술들은 반응형 방식(사고 발생 후 대응)에 머무르며, 능동적 사고 예방이 어렵습니다. 특히 비상정지(Emergency Stop) 방식은 현장에서 빈번한 해제로 인해 사고를 오히려 증가시키는 문제가 있습니다. SeRT 플랫폼은 EEG 기반 인간 피드백을 통합하여 Robotic Transformer를 안전 강화함으로써, 인간과 로봇이 충돌 없이 효율적으로 협업할 수 있는 미래를 실현합니다.

### Problems to Solve (해결하려는 문제)
1. **산업 현안: 협동 로봇 사고 급증**
   - ISO/TS 15066:2016 표준 준수만으로는 부족하며, 실제 현장에서 비상정지 해제가 빈번하여 사고가 증가
   - 단순한 비상정지 방식으로는 신속한 사고 예방이 어려움

2. **기술적 한계: 자동 계획의 지연 문제**
   - 역기구학 기반 자동 계획 시스템은 계산 지연으로 신속한 회피 불가능
   - 강화학습은 보상 함수 설계의 어려움으로 인간 행동 변화에 대응하기 힘듦

3. **AI 모델 한계: Robotic Transformer의 안전 메커니즘 부족**
   - RT-1/RT-2 등 최신 모델은 로봇 단독 작업에 최적화되어 인간 협업 시 위험 판단 및 회피 메커니즘이 없음
   - VLA(비전-언어-행동) 모델에 안전 행동 강화가 필요

4. **데이터 부족: 인간-로봇 협업 학습 데이터 부재**
   - 실제 인간-로봇 충돌 상황 반복 재현이 불가능하여 대규모 학습 데이터 확보 어려움
   - EEG 기반 암묵적 피드백 데이터셋이 부족

### Product Description
SeRT 플랫폼은 위 문제들을 해결하기 위해 햅틱 장비를 결합한 인간-로봇 협업 시뮬레이션 환경에서 EEG 데이터를 수집하고, 이를 Robotic Transformer에 통합하여 안전 행동을 학습하는 종합 플랫폼입니다. 연구실적[9-11]을 기반으로 IsaacSim 시뮬레이션과 실제 햅틱 장비를 연동하여, 인간의 뇌파 신호를 실시간으로 분석하고 위험 상황을 예측하여 능동적으로 회피하는 AI를 개발합니다.

### Core Features
- **데이터 수집 모듈**: CoppeliaSim 기반 인간-인간/인간-로봇 협업 데모 데이터 생성 (EEG + 햅틱 데이터 통합)
- **EEG 처리 모듈**: 암묵적 Human Feedback 분석 및 위험 감지 (안전/위험 구간 매핑)
- **학습 플랫폼**: Safety-enhanced Robotic Transformer Fine-Tuning (EEG 피드백 입력으로 안전 행동 강화)
- **평가 모듈**: Calvin Benchmark Test 및 충돌회피 시뮬레이션 (SOTA 78.4% 상회 목표)

### Technology Stack
- **시뮬레이션**: IsaacSim (멀티에이전트 환경 구축)
- **AI/ML**: PyTorch, Robotic Transformer (RT-1, RT-2 기반 Fine-Tune)
- **EEG 처리**: MNE, PyEEG (신호 전처리 및 특징 추출)
- **햅틱 장비**: Meta Quest 3, bHaptics TactGlove DK2 (실시간 협업 데이터 수집)

---

## 3. Target Users

### Primary Users
- **연구자**: 컴퓨터공학과/로봇공학 연구팀
- **개발자**: AI/로봇 소프트웨어 엔지니어
- **산업 사용자**: 협동 로봇 제조사 (현대, LG 등)

### User Personas
1. **Dr. Kim (연구자)**: 인간-로봇 협업 안전성 연구에 관심, EEG 데이터 분석 필요
2. **Engineer Park (개발자)**: Robotic Transformer 구현 및 Fine-Tuning 담당
3. **Manager Lee (산업)**: 협동 로봇 안전 표준 준수를 위한 솔루션 요구

---

## 4. Requirements

### Business Requirements
- BR1: 2026년 SCIE 저널 게재
- BR2: 산업 파트너와 협업 가능성 확보
- BR3: 오픈소스 공개로 학술 커뮤니티 기여

### Functional Requirements
- FR1: IsaacSim 환경에서 인간-로봇 협업 시뮬레이션 실행
- FR2: 햅틱 장비로부터 실시간 데이터 수집
- FR3: EEG 신호로부터 위험 감지 및 피드백 생성
- FR4: Robotic Transformer에 Safety 메커니즘 통합
- FR5: Calvin Benchmark Test 수행 및 평가

### Non-Functional Requirements
- NFR1: 실시간 성능 (EEG 처리 < 100ms)
- NFR2: 확장성 (다양한 로봇 모델 지원)
- NFR3: 정확도 (충돌회피 85% 이상)
- NFR4: 사용성 (Python API 제공)
- NFR5: 보안 (데이터 프라이버시 보호)

---

## 5. User Stories

### Epic 1: 데이터 수집 및 플랫폼 구축
- **US1**: 연구자로서, CoppeliaSim에서 인간-인간 협업 데모를 실행하여 EEG 데이터를 수집하고 싶다.
- **US2**: 개발자로서, 햅틱 장비를 연동하여 실시간 피드백을 받을 수 있는 인터페이스가 필요하다.

### Epic 2: AI 모델 개발
- **US3**: 연구자로서, EEG 기반 피드백을 Robotic Transformer에 통합하여 Safety-enhanced 모델을 학습시키고 싶다.
- **US4**: 개발자로서, 공개 데이터셋(RT-1, RT-2)을 활용한 Fine-Tuning 기능을 구현하고 싶다.

### Epic 3: 평가 및 검증
- **US5**: 연구자로서, Calvin Benchmark Test에서 80% 성공률을 달성하여 SOTA를 상회하는 결과를 얻고 싶다.
- **US6**: 산업 사용자로서, 실제 로봇 환경에서 충돌회피 성능을 검증할 수 있는 도구가 필요하다.

---

## 6. Acceptance Criteria

### AC1: 데이터 수집 모듈
- IsaacSim에서 100개 이상의 협업 시나리오 실행 가능
- EEG 데이터 샘플링 레이트 256Hz 이상
- 햅틱 장비 연동 성공률 95%

### AC2: 학습 플랫폼
- Robotic Transformer 학습 시간 < 24시간 (GPU 기준)
- Safety 메커니즘 통합으로 충돌회피 정확도 85% 달성
- 공개 데이터셋 활용으로 재현성 확보

### AC3: 평가 모듈
- Calvin Benchmark Test 자동 실행 및 결과 보고
- 시각화 대시보드 제공 (성공률 그래프 등)

---

## 7. Design Considerations

### UI/UX
- 웹 기반 대시보드 (Streamlit or Gradio)
- 시뮬레이션 영상 스트리밍
- 실시간 EEG 신호 모니터링

### Architecture
- 모듈형 설계 (데이터 수집, 처리, 학습, 평가 분리)
- 클라우드 지원 (Google Colab, AWS)
- API 기반 확장성

### Data Flow
1. 햅틱 장비 → CoppeliaSim → EEG 수집
2. 데이터 전처리 → 피드백 생성
3. Robotic Transformer 학습 → Safety 통합
4. 평가 실행 → 결과 분석

---

## 8. Work Breakdown Structure (WBS)

### 프로젝트: SeRT 학습 플랫폼 개발

```
SeRT 학습 플랫폼 개발
├── 1. 기획 및 설계 (2025년 10월 - 12월)
│   ├── 1.1 요구사항 정의
│   │   ├── 1.1.1 기능 요구사항 수집
│   │   ├── 1.1.2 비기능 요구사항 정의
│   │   └── 1.1.3 PRD 작성 및 승인
│   ├── 1.2 기술 스택 선정
│   │   ├── 1.2.1 CoppeliaSim 환경 검토
│   │   ├── 1.2.2 Robotic Transformer 모델 선택
│   │   └── 1.2.3 햅틱 장비 호환성 테스트
│   ├── 1.3 아키텍처 설계
│   │   ├── 1.3.1 시스템 아키텍처 다이어그램
│   │   ├── 1.3.2 데이터 플로우 설계
│   │   └── 1.3.3 API 인터페이스 정의
│   └── 1.4 프로토타입 설계
│       ├── 1.4.1 모듈별 프로토타입 구현
│       └── 1.4.2 초기 테스트 및 검증
├── 2. 개발 (2026년 1월 - 6월)
│   ├── 2.1 데이터 수집 모듈 개발
│   │   ├── 2.1.1 CoppeliaSim 시뮬레이션 환경 구축
│   │   ├── 2.1.2 햅틱 장비 연동 시스템
│   │   ├── 2.1.3 EEG 데이터 수집 인터페이스
│   │   └── 2.1.4 데모 데이터 생성 파이프라인
│   ├── 2.2 EEG 처리 모듈 개발
│   │   ├── 2.2.1 신호 전처리 알고리즘
│   │   ├── 2.2.2 특징 추출 및 피드백 생성
│   │   ├── 2.2.3 위험 감지 모델 구현
│   │   └── 2.2.4 실시간 처리 최적화
│   ├── 2.3 학습 플랫폼 개발
│   │   ├── 2.3.1 Robotic Transformer 모델 로드
│   │   ├── 2.3.2 Safety 메커니즘 통합
│   │   ├── 2.3.3 Fine-Tuning 파이프라인
│   │   └── 2.3.4 공개 데이터셋 통합
│   └── 2.4 평가 모듈 개발
│       ├── 2.4.1 Calvin Benchmark Test 연동
│       ├── 2.4.2 충돌회피 평가 지표 구현
│       ├── 2.4.3 시각화 대시보드
│       └── 2.4.4 자동화된 평가 스크립트
├── 3. 테스트 및 평가 (2026년 7월 - 9월)
│   ├── 3.1 단위 테스트
│   │   ├── 3.1.1 각 모듈별 기능 테스트
│   │   └── 3.1.2 버그 수정 및 리팩토링
│   ├── 3.2 통합 테스트
│   │   ├── 3.2.1 전체 시스템 통합 테스트
│   │   └── 3.2.2 성능 및 안정성 검증
│   ├── 3.3 성능 평가
│   │   ├── 3.3.1 Calvin Benchmark Test 수행
│   │   ├── 3.3.2 충돌회피 성공률 측정
│   │   └── 3.3.3 EEG 피드백 정확도 평가
│   └── 3.4 최적화
│       ├── 3.4.1 모델 성능 튜닝
│       └── 3.4.2 시스템 리소스 최적화
└── 4. 배포 및 유지보수 (2026년 10월 - 12월)
    ├── 4.1 오픈소스 공개
    │   ├── 4.1.1 코드 정리 및 문서화
    │   ├── 4.1.2 GitHub 리포지토리 설정
    │   └── 4.1.3 라이선스 및 기여 가이드 작성
    ├── 4.2 논문 투고
    │   ├── 4.2.1 결과 분석 및 논문 작성
    │   ├── 4.2.2 SCIE 저널 선정 및 투고
    │   └── 4.2.3 피어 리뷰 대응
    └── 4.3 유지보수 및 확장
        ├── 4.3.1 사용자 피드백 수집
        ├── 4.3.2 버그 수정 및 업데이트
        └── 4.3.3 추가 기능 개발 계획
```

### WBS 사용 방법
- 각 작업 항목에 담당자, 예상 시간, 의존 관계를 할당
- 진행 상황을 주기적으로 업데이트하여 프로젝트 추적
- 변경 사항 발생 시 WBS 재구성

---

## 9. Timeline

### Phase 1: 기획 및 설계 (2025년 10월 - 12월)
- 요구사항 정의 및 PRD 작성
- 기술 스택 선정 및 프로토타입 설계

### Phase 2: 개발 (2026년 1월 - 6월)
- 데이터 수집 모듈 구현
- EEG 처리 및 학습 플랫폼 개발
- 햅틱 장비 연동

### Phase 3: 테스트 및 평가 (2026년 7월 - 9월)
- Calvin Benchmark Test 수행
- 충돌회피 성능 검증
- 버그 수정 및 최적화

### Phase 4: 배포 및 논문 (2026년 10월 - 12월)
- 오픈소스 공개
- SCIE 저널 투고
- 산업 파트너 데모

---

## 9. Risks and Assumptions

### Risks
- **R1**: EEG 데이터 품질 저하 → 완충 데이터셋 확보
- **R2**: 햅틱 장비 호환성 문제 → 사전 테스트 강화
- **R3**: Calvin Benchmark 목표 미달성 → 추가 Fine-Tuning

### Assumptions
- **A1**: 공개 데이터셋(RT-1, RT-2) 접근 가능
- **A2**: GPU 리소스 충분히 확보
- **A3**: 윤리적 승인 및 데이터 프라이버시 준수

---

## 10. Appendices

### Appendix A: Reference Documents
- 연구계획서: S2_자연_4.컴퓨터공학과 신청서.pdf
- 관련 논문: RT-1, RT-2, Calvin Benchmark 등

### Appendix B: Team
- 책임자: 김현석
- 참여자: 학부생 2명, 대학원생 1명