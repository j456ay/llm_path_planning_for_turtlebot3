#!/bin/bash

echo "========================================="
echo "LLM Path Planning 의존성 설치 스크립트"
echo "========================================="

# Python 의존성 설치
echo "1. Python 의존성 설치..."
pip3 install requests numpy setuptools

# Ollama 설치 확인
echo "2. Ollama 설치 확인..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama가 이미 설치되어 있습니다."
else
    echo "Ollama 설치 중..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# LLM 모델 다운로드
echo "3. LLM 모델 다운로드..."
if ollama list | grep -q "llama3.1:8b"; then
    echo "✓ llama3.1:8b 모델이 이미 다운로드되어 있습니다."
else
    echo "llama3.1:8b 모델 다운로드 중... (시간이 오래 걸릴 수 있습니다)"
    ollama pull llama3.1:8b
fi

# ROS2 의존성 확인
echo "4. ROS2 의존성 확인..."
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠ ROS2 환경이 설정되지 않았습니다."
    echo "다음 명령을 실행하세요:"
    echo "source /opt/ros/humble/setup.bash"
else
    echo "✓ ROS2 $ROS_DISTRO 환경이 설정되어 있습니다."
fi

# 추가 ROS2 패키지 설치
echo "5. 추가 ROS2 패키지 설치..."
sudo apt update
sudo apt install -y ros-$ROS_DISTRO-navigation2 ros-$ROS_DISTRO-nav2-bringup \
                   ros-$ROS_DISTRO-turtlebot3 ros-$ROS_DISTRO-turtlebot3-simulations

echo ""
echo "========================================="
echo "설치가 완료되었습니다!"
echo "========================================="
echo ""
echo "다음 단계:"
echo "1. 터미널에서 'ollama serve' 실행 (백그라운드)"
echo "2. ROS2 워크스페이스에서 'colcon build' 실행"
echo "3. TurtleBot3 시뮬레이션 또는 실제 로봇 실행"
echo "4. Navigation2 실행"
echo "5. LLM Path Planner 실행"
