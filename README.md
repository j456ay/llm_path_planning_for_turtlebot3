# LLM Path Planning for TurtleBot3

TurtleBot3 Waffle Pi + ROS2 Humble 환경에서 LLM을 활용한 경로 계획 패키지입니다.

## 개요

이 패키지는 LLM(Large Language Model)을 사용하여 여러 개의 경로 후보를 생성하고, 이를 평가하여 최적의 경로를 선택하는 시스템입니다. 실제 경로 실행은 Nav2가 담당하며, LLM은 창의적이고 다양한 경로 후보 생성에 집중합니다.

## 주요 특징

- **LLM 기반 경로 후보 생성**: Ollama를 통해 로컬 LLM 서버와 통신하여 다양한 경로 후보 생성
- **충돌 검증 및 점수화**: OccupancyGrid 맵을 기반으로 각 후보 경로를 평가
- **실시간 시각화**: RViz2에서 후보 경로들과 선택된 최적 경로를 시각화
- **파라미터 기반 설정**: YAML 파일로 모든 설정값 조정 가능

## 시스템 요구사항

### 하드웨어
- TurtleBot3 Waffle Pi
- Raspberry Pi 4
- RealSense 카메라 (현재는 SLAM/맵 구축용)

### 소프트웨어
- Ubuntu 22.04
- ROS2 Humble
- Nav2
- Ollama (로컬 LLM 서버)
- Python 3.10+

## 설치 및 빌드

### 1. 의존성 설치

```bash
# ROS2 Humble과 Nav2가 이미 설치되어 있다고 가정
sudo apt update
sudo apt install python3-pip

# Python 의존성
pip3 install requests numpy
```

### 2. Ollama 설치 및 설정

```bash
# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# LLM 모델 다운로드 (예: Llama 3.1 8B)
ollama pull llama3.1:8b

# Ollama 서버 실행
ollama serve
```

### 3. 패키지 빌드

```bash
# ROS2 워크스페이스로 이동
cd ~/ros2_ws/src

# 패키지 복사 (이 디렉토리 구조를 src 폴더에 복사)
# llm_path_planning/ 폴더를 ~/ros2_ws/src/에 위치

# 빌드
cd ~/ros2_ws
colcon build --packages-select llm_path_planning
source install/setup.bash
```

## 사용법

### 1. Ollama 서버 실행

```bash
ollama serve
```

### 2. TurtleBot3 기본 실행

```bash
# 시뮬레이션 환경 (Gazebo)
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 또는 실제 로봇
ros2 launch turtlebot3_bringup robot.launch.py
```

### 3. SLAM 및 Navigation 실행

```bash
# SLAM
ros2 launch turtlebot3_cartographer cartographer.launch.py

# Navigation (맵이 준비된 후)
ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=saved_map.yaml
```

### 4. LLM Path Planning 실행

```bash
# 기본 실행
ros2 launch llm_path_planning llm_path_planning_launch.py

# 사용자 정의 파라미터로 실행
ros2 launch llm_path_planning llm_path_planning_launch.py \
  params_file:=/path/to/your/params.yaml
```

### 5. RViz2 실행 및 목표점 설정

```bash
# RViz2 실행 (사전 구성된 설정 사용)
rviz2 -d install/llm_path_planning/share/llm_path_planning/rviz/llm_path_planning.rviz

# 또는 기본 RViz2
rviz2
```

RViz2에서:
1. "2D Nav Goal" 도구를 사용하여 목표점 설정
2. 후보 경로들이 다양한 색상으로 표시됨
3. 선택된 최적 경로가 cyan 색상으로 표시됨

## 토픽 정보

### 구독 토픽
- `/map` (nav_msgs/OccupancyGrid): 환경 맵
- `/amcl_pose` (geometry_msgs/PoseWithCovarianceStamped): 현재 로봇 위치
- `/goal_pose` (geometry_msgs/PoseStamped): 목표 위치

### 발행 토픽  
- `/candidate_paths` (visualization_msgs/MarkerArray): 후보 경로들 시각화
- `/selected_path` (nav_msgs/Path): 선택된 최적 경로

## 설정 파라미터

주요 파라미터들은 `config/params.yaml`에서 수정할 수 있습니다:

```yaml
llm_path_planner:
  ros__parameters:
    # Ollama 설정
    ollama_model: "llama3.1:8b"
    num_candidates: 5
    
    # 로봇 설정
    robot_radius: 0.2
    safety_margin: 0.1
    
    # 평가 가중치
    weights:
      collision_penalty: 1000.0
      path_length: 1.0
      smoothness: 0.5
      clearance: 2.0
```

## 시스템 구조

```
LLM Path Planning System
├── Map Processor: OccupancyGrid 처리 및 다운샘플링
├── Ollama Client: LLM API 통신 및 프롬프트 관리
├── Path Evaluator: 후보 경로 평가 및 점수화
└── Main Node: ROS2 노드 및 전체 워크플로우 관리
```

## 워크플로우

1. **맵 및 위치 정보 수집**: `/map`, `/amcl_pose`, `/goal_pose` 구독
2. **맵 전처리**: 고해상도 맵을 저해상도로 다운샘플링
3. **LLM 쿼리**: 맵 정보와 시작/목표점을 LLM에 전달하여 후보 경로 생성
4. **경로 평가**: 각 후보를 충돌, 길이, 부드러움, 클리어런스 기준으로 평가
5. **최적 경로 선택**: 가중합 점수가 가장 낮은 후보 선택
6. **시각화**: RViz2에서 모든 후보와 선택된 경로 표시

## 문제 해결

### Ollama 연결 오류
```bash
# Ollama 서버 상태 확인
curl http://localhost:11434/api/tags

# 모델 목록 확인
ollama list
```

### ROS2 토픽 확인
```bash
# 발행되는 토픽 확인
ros2 topic list
ros2 topic echo /candidate_paths
ros2 topic echo /selected_path
```

### 로그 확인
```bash
# 노드 로그 확인
ros2 launch llm_path_planning llm_path_planning_launch.py --ros-args --log-level DEBUG
```

## 확장 계획

- [ ] 동적 장애물 감지 (YOLO 기반 사람 인식)
- [ ] 더 정교한 경로 평가 알고리즘
- [ ] 다양한 LLM 모델 지원
- [ ] 실시간 경로 재계획
- [ ] 사용자 선호도 학습

## 기여

이 패키지는 연구 목적으로 개발되었습니다. 이슈나 개선사항이 있으시면 언제든 연락해 주세요.

## 라이선스

Apache 2.0 License
