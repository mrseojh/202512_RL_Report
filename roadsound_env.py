"""
RoadSoundEnv: 도로 주행음 기반 Gym 스타일 RL 환경

버전: 1.0.0
날짜: 2025-01-XX
변경 이력:
  - 1.0.0: 초기 MVP 구현

모듈 개요:
  - 목적: Jetson Orin Nano에서 실행 가능한 단일 파일 RL 환경
  - 입력: 사전 전처리된 에너지 bin 시퀀스(정수 0~4의 1차원 numpy 배열)
  - 동작: 1초 단위 MDP 스텝, 행동 0(Skip), 1(Record)
  - 보상: 이벤트(에너지 bin>=3) 탐지 성과에 따라 지급
  - 데모: 고정 시드 기반 무작위 정책으로 최대 10스텝 실행
"""

import numpy as np
from typing import Optional, Any, Dict, Tuple

# 보상 및 임계값 상수
EVENT_THRESHOLD = 3  # 이벤트 임계값 (s_t >= 3이면 이벤트)
REWARD_TP = 1.0      # True Positive: event=1 & action=1
REWARD_FN = -1.0     # False Negative: event=1 & action=0
REWARD_FP = -0.2     # False Positive: event=0 & action=1
REWARD_TN = 0.2      # True Negative: event=0 & action=0

# 조건부 gymnasium import
try:
    import gymnasium as gym
    BaseEnv = gym.Env
except ImportError:
    # Gymnasium이 없을 경우 최소 커스텀 베이스 클래스
    class BaseEnv:
        """
        최소 Env 베이스 클래스 (gymnasium 부재 시 사용)
        
        Gym 없이도 실행 가능하도록 하는 최소 인터페이스 제공.
        하위 클래스에서 reset과 step 메서드를 구현해야 함.
        """
        def reset(self) -> int:
            """환경을 초기 상태로 리셋하고 초기 상태를 반환"""
            raise NotImplementedError
        
        def step(self, action: int) -> Tuple[Optional[int], float, bool, Dict[str, Any]]:
            """
            행동을 수행하고 다음 상태, 보상, 종료 여부, 정보를 반환
            
            Args:
                action: 행동 (0: Skip, 1: Record)
            
            Returns:
                (next_state, reward, done, info) 튜플
            """
            raise NotImplementedError


def _warn(msg: str) -> None:
    """경고 메시지 출력 헬퍼"""
    print(f"[RoadSoundEnv][WARN] {msg}")


def _error(msg: str) -> None:
    """오류 메시지 출력 헬퍼"""
    print(f"[RoadSoundEnv][ERROR] {msg}")


def load_energy_bins(path: str) -> np.ndarray:
    """
    NPY 파일에서 에너지 bin 배열을 로드하고 유효성 검증/보정을 수행
    
    Args:
        path: NPY 파일 경로
    
    Returns:
        1차원 정수 배열, dtype=int, shape=(T,), 값 범위 [0, 4]
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 빈 배열이거나 캐스팅 실패 시
    """
    # 파일 로드
    arr = np.load(path)
    
    # 1차원 검사 및 평탄화
    if arr.ndim != 1:
        _warn(f"입력 배열이 1차원이 아닙니다 (shape={arr.shape}). 1차원으로 평탄화합니다.")
        arr = arr.ravel()
    
    # dtype 정수형 검사 및 캐스팅
    if not np.issubdtype(arr.dtype, np.integer):
        _warn(f"입력 배열이 정수형이 아닙니다 (dtype={arr.dtype}). 정수로 캐스팅합니다.")
        try:
            arr = arr.astype(int)
        except (ValueError, TypeError) as e:
            _error(f"정수로 캐스팅할 수 없습니다: {e}")
            raise ValueError(f"정수로 캐스팅 실패: {e}")
    
    # 값 범위 검사 및 클리핑
    if arr.size > 0:
        min_val = np.min(arr)
        max_val = np.max(arr)
        if min_val < 0 or max_val > 4:
            _warn(f"값 범위가 [0, 4]를 벗어났습니다 (min={min_val}, max={max_val}). 클리핑합니다.")
            arr = np.clip(arr, 0, 4)
    
    # 빈 배열 검사
    if len(arr) == 0:
        _error("빈 배열입니다. 최소 길이 1이 필요합니다.")
        raise ValueError("빈 배열은 허용되지 않습니다.")
    
    return arr


class RoadSoundEnv(BaseEnv):
    """
    도로 주행음 기반 RL 환경
    
    에너지 bin 시퀀스를 입력으로 받아 1초 단위 MDP 스텝을 수행합니다.
    행동에 따라 이벤트(에너지 bin>=3) 탐지 성과를 보상으로 환산합니다.
    """
    
    def __init__(self, energy_bins: np.ndarray) -> None:
        """
        환경 초기화
        
        Args:
            energy_bins: 0~4 사이의 정수로 이루어진 1차원 numpy 배열
        
        Raises:
            ValueError: 빈 배열이거나 유효성 검증 실패 시
        """
        # 입력 검증 및 보정 (load_energy_bins와 동일한 로직)
        if energy_bins.ndim != 1:
            _warn(f"입력 배열이 1차원이 아닙니다 (shape={energy_bins.shape}). 1차원으로 평탄화합니다.")
            energy_bins = energy_bins.ravel()
        
        if not np.issubdtype(energy_bins.dtype, np.integer):
            _warn(f"입력 배열이 정수형이 아닙니다 (dtype={energy_bins.dtype}). 정수로 캐스팅합니다.")
            try:
                energy_bins = energy_bins.astype(int)
            except (ValueError, TypeError) as e:
                _error(f"정수로 캐스팅할 수 없습니다: {e}")
                raise ValueError(f"정수로 캐스팅 실패: {e}")
        
        if energy_bins.size > 0:
            min_val = np.min(energy_bins)
            max_val = np.max(energy_bins)
            if min_val < 0 or max_val > 4:
                _warn(f"값 범위가 [0, 4]를 벗어났습니다 (min={min_val}, max={max_val}). 클리핑합니다.")
                energy_bins = np.clip(energy_bins, 0, 4)
        
        if len(energy_bins) == 0:
            _error("빈 배열입니다. 최소 길이 1이 필요합니다.")
            raise ValueError("빈 배열은 허용되지 않습니다.")
        
        # 상태 저장
        self.energy_bins = energy_bins
        self.n_states = 5
        self.n_actions = 2
        self.t = 0
        self._last_state: Optional[int] = None
    
    def reset(self) -> int:
        """
        환경을 초기 상태로 리셋
        
        Returns:
            초기 상태 s_0 (정수)
        """
        self.t = 0
        s0 = int(self.energy_bins[0])
        self._last_state = s0
        return s0
    
    def step(self, action: int) -> Tuple[Optional[int], float, bool, Dict[str, Any]]:
        """
        행동을 수행하고 다음 상태, 보상, 종료 여부, 정보를 반환
        
        Args:
            action: 행동 (0: Skip, 1: Record). 범위 밖 값은 보정됨
        
        Returns:
            (next_state, reward, done, info) 튜플
            - next_state: 다음 상태 (정수) 또는 None (종료 시)
            - reward: 보상 (float)
            - done: 에피소드 종료 여부 (bool)
            - info: 정보 딕셔너리 (t, event, action, reward)
        """
        # 행동 보정
        if action not in {0, 1}:
            _warn(f"행동이 {0, 1} 범위를 벗어났습니다 (action={action}). 보정합니다.")
            action = int(bool(action))
        
        # 현재 상태
        s_t = int(self.energy_bins[self.t])
        
        # 이벤트 라벨 계산
        event = 1 if s_t >= EVENT_THRESHOLD else 0
        
        # 보상 계산 (보상 테이블)
        if event == 1 and action == 1:
            r = REWARD_TP
        elif event == 1 and action == 0:
            r = REWARD_FN
        elif event == 0 and action == 1:
            r = REWARD_FP
        else:  # event == 0 and action == 0
            r = REWARD_TN
        
        # 시간 진행
        t_before = self.t
        self.t += 1
        
        # 종료 판정 및 다음 상태 계산
        if self.t >= len(self.energy_bins):
            done = True
            next_state = None
        else:
            done = False
            next_state = int(self.energy_bins[self.t])
        
        # 마지막 상태 업데이트
        self._last_state = next_state if next_state is not None else s_t
        
        # info 딕셔너리 구성
        info: Dict[str, Any] = {
            "t": t_before,
            "event": event,
            "action": int(action),
            "reward": float(r),
        }
        
        return (next_state, r, done, info)


if __name__ == "__main__":
    # 고정 시드 설정
    rng = np.random.default_rng(42)
    
    # 데이터 로드 시도
    try:
        train_bins = load_energy_bins("train_bins.npy")
    except FileNotFoundError:
        _warn("train_bins.npy 파일을 찾을 수 없습니다. 합성 데이터를 생성합니다.")
        train_bins = rng.integers(0, 5, size=20, dtype=int)
    
    # 환경 생성
    env = RoadSoundEnv(train_bins)
    
    # 리셋 및 초기 상태
    s = env.reset()
    
    # 데모 루프 (최대 10스텝 또는 done=True까지)
    max_steps = 10
    step_count = 0
    
    print("=== RoadSoundEnv 데모 시작 ===")
    while step_count < max_steps:
        # 무작위 행동 선택
        action = rng.integers(0, 2)
        
        # 스텝 수행
        next_state, reward, done, info = env.step(action)
        
        # 출력
        print(f"t={info['t']}, s={s}, a={action}, r={reward:.2f}, done={done}, info={info}")
        
        # 다음 상태 업데이트
        s = next_state if next_state is not None else s
        
        step_count += 1
        
        # 종료 확인
        if done:
            break
    
    print("=== 데모 종료 ===")

