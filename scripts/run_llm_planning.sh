#!/bin/bash

# LLM Path Planning 실행 스크립트

echo "========================================="
echo "LLM Path Planning for TurtleBot3 시작"
echo "========================================="

# Ollama 서버 상태 확인
echo "1. Ollama 서버 상태 확인..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✓ Ollama 서버가 실행중입니다."
else
    echo "✗ Ollama 서버가 실행되지 않았습니다."
    echo "다음 명령으로 Ollama를 실행하세요: ollama serve"
    exit 1
fi

# 모델 확인
echo "2. LLM 모델 확인..."
if ollama list | grep -q "llama3.1:8b"; then
    echo "✓ llama3.1:8b 모델이 설치되어 있습니다."
else
    echo "⚠ llama3.1:8b 모델이 설치되지 않았습니다."
    echo "다음 명령으로 모델을 다운로드하세요: ollama pull llama3.1:8b"
fi

# ROS2 환경 확인
echo "3. ROS2 환경 확인..."
if [ -z "$ROS_DISTRO" ]; then
    echo "✗ ROS2 환경이 설정되지 않았습니다."
    echo "source /opt/ros/humble/setup.bash를 실행하세요."
    exit 1
else
    echo "✓ ROS2 $ROS_DISTRO 환경이 설정되어 있습니다."
fi

# 워크스페이스 소싱
if [ -f "install/setup.bash" ]; then
    echo "4. 워크스페이스 소싱..."
    source install/setup.bash
    echo "✓ 워크스페이스가 소싱되었습니다."
else
    echo "✗ install/setup.bash 파일을 찾을 수 없습니다."
    echo "colcon build를 먼저 실행하세요."
    exit 1
fi

# 필수 토픽 확인
echo "5. 필수 토픽 확인..."
timeout 3s ros2 topic list > /tmp/topics.txt 2>/dev/null

if grep -q "/map" /tmp/topics.txt; then
    echo "✓ /map 토픽이 발행중입니다."
else
    echo "⚠ /map 토픽이 발행되지 않습니다. Navigation을 먼저 실행하세요."
fi

if grep -q "/amcl_pose" /tmp/topics.txt; then
    echo "✓ /amcl_pose 토픽이 발행중입니다."
else
    echo "⚠ /amcl_pose 토픽이 발행되지 않습니다. AMCL을 먼저 실행하세요."
fi

echo ""
echo "========================================="
echo "LLM Path Planner 노드 실행"
echo "========================================="

# LLM Path Planning 실행
ros2 launch llm_path_planning llm_path_planning_launch.py

echo ""
echo "LLM Path Planning이 종료되었습니다."
