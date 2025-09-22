#!/usr/bin/env python3

import os
import yaml
import json
import requests
import cv2
import numpy as np
from PIL import Image
from skimage.feature import peak_local_max
from scipy import ndimage
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import tf_transformations
import time
import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

# YOLO 관련 import (ultralytics 사용)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO를 사용하려면 'pip install ultralytics' 설치가 필요합니다.")
    YOLO_AVAILABLE = False

# Raspberry Pi 카메라 관련 import
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
    USE_LEGACY_PICAMERA = False
except ImportError:
    try:
        # 구버전 picamera 라이브러리 지원
        import picamera
        import picamera.array
        PI_CAMERA_AVAILABLE = True
        USE_LEGACY_PICAMERA = True
    except ImportError:
        print("⚠️ Pi 카메라를 사용하려면 'sudo apt install python3-picamera2' 또는 'pip install picamera' 설치가 필요합니다.")
        PI_CAMERA_AVAILABLE = False
        USE_LEGACY_PICAMERA = False

# ----------------------------------------------------------------------------------
# 데이터 클래스 정의
# ----------------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """검출 결과를 담는 데이터 클래스"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    timestamp: float

# ----------------------------------------------------------------------------------
# LLM #1: 지도 분석 전문가 클래스 (기존과 동일)
# ----------------------------------------------------------------------------------
class IntelligentRoomMerger:
    def __init__(self, ollama_url="http://localhost:11434/api/chat", model_name="llama3:latest"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.map_metadata = None
        self.map_data = None

    def load_map_data(self, yaml_path: str) -> bool:
        try:
            expanded_path = os.path.expanduser(yaml_path)
            with open(expanded_path, 'r') as file:
                self.map_metadata = yaml.safe_load(file)
            yaml_dir = os.path.dirname(expanded_path)
            pgm_path = os.path.join(yaml_dir, self.map_metadata['image'])
            if not os.path.exists(pgm_path):
                print(f"오류: PGM 파일 '{pgm_path}'을 찾을 수 없습니다.")
                return False
            with Image.open(pgm_path) as img:
                self.map_data = np.array(img.convert('L'))
                self.map_metadata['width'] = img.width
                self.map_metadata['height'] = img.height
            print("맵 데이터 로드 성공.")
            return True
        except Exception as e:
            print(f"맵 데이터 로드 중 오류 발생: {e}")
            return False

    def find_candidate_rooms(self) -> list:
        print("1단계 (CV): 맵에서 모든 방 후보를 과탐지합니다...")
        _, binary_map = cv2.threshold(self.map_data, 253, 255, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
        coordinates = peak_local_max(dist_transform, min_distance=15, threshold_abs=0.1 * dist_transform.max())
        markers = np.zeros(dist_transform.shape, dtype=bool)
        markers[tuple(coordinates.T)] = True
        markers, _ = ndimage.label(markers)
        labels = cv2.watershed(cv2.cvtColor(self.map_data, cv2.COLOR_GRAY2BGR), markers)
        candidates = []
        for label in np.unique(labels):
            if label <= 0: continue
            mask = (labels == label).astype(np.uint8) * 255
            area = cv2.countNonZero(mask)
            if area < 100: continue
            x, y, w, h = cv2.boundingRect(mask)
            candidates.append({"id": int(label), "area": int(area), "bbox_pixels": [x, y, x + w, y + h]})
        print(f"총 {len(candidates)}개의 방 후보를 찾았습니다.")
        return candidates

    def reason_and_merge_rooms_with_llm(self, candidates: list) -> dict:
        print("2단계 (LLM #1 - 지도 분석): 후보 목록을 분석하여 최종 6개 방을 결정합니다...")
        prompt = f"""You are an intelligent robotic space analysis agent. Your task is to analyze a list of room 'candidates' and clean this list to identify exactly 6 rooms. The list is: {json.dumps(candidates, indent=2)}.
Rules:
1. The largest candidate is 'outside' and must be ignored.
2. Small candidates inside or very close to larger ones are fragments and should be merged.
3. The final list MUST contain exactly 6 rooms: 2 large 'main rooms' and 4 smaller 'sub-rooms'.
Return ONLY a JSON object in the format: {{"final_rooms": {{"main_rooms": [{{"id": "<ID>"}}, {{"id": "<ID>"}}], "sub_rooms": [{{"id": "<ID>"}}, {{"id": "<ID>"}}, {{"id": "<ID>"}}, {{"id": "<ID>"}}]}}}}"""
        try:
            payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "stream": False, "format": "json"}
            response = requests.post(self.ollama_url, json=payload, timeout=300)
            response.raise_for_status()
            return json.loads(response.json().get('message', {}).get('content', '{}'))
        except Exception as e:
            print(f"LLM 지도 분석 실패: {e}"); return {}

    def pixel_to_world(self, pixel_coords: tuple) -> tuple:
        x_pixel, y_pixel = pixel_coords
        resolution = self.map_metadata['resolution']
        origin = self.map_metadata['origin']
        map_height = self.map_metadata['height']
        x_world = origin[0] + x_pixel * resolution
        y_world = origin[1] + (map_height - y_pixel) * resolution
        return (x_world, y_world)

    def get_room_database(self, yaml_path: str):
        if not self.load_map_data(yaml_path): return None
        candidates = self.find_candidate_rooms()
        if not candidates: return None
        final_decision = self.reason_and_merge_rooms_with_llm(candidates)
        candidate_lookup = {c['id']: c for c in candidates}
        final_room_ids = [room.get('id') for room_type in ["main_rooms", "sub_rooms"] for room in final_decision.get("final_rooms", {}).get(room_type, [])]
        if not final_room_ids or None in final_room_ids:
            print("LLM의 결정에서 유효한 방 ID를 추출하지 못했습니다."); return None
        final_rooms_details = []
        for room_id in final_room_ids:
            if candidate := candidate_lookup.get(int(room_id)):
                bbox = candidate['bbox_pixels']
                center_x_pix, center_y_pix = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                final_rooms_details.append({"id": room_id, "world_coords": self.pixel_to_world((center_x_pix, center_y_pix)), "area": candidate['area']})
        sorted_rooms = sorted(final_rooms_details, key=lambda x: x['area'], reverse=True)
        room_database = {chr(ord('A') + i): room['world_coords'] for i, room in enumerate(sorted_rooms)}
        print("\n=== 최종 방 데이터베이스 생성 완료 ===\n" + json.dumps(room_database, indent=2))
        return room_database

# ----------------------------------------------------------------------------------
# 🚨 NEW: YOLO 기반 실시간 물체/사람 검출 클래스
# ----------------------------------------------------------------------------------
class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5, camera_resolution=(640, 480)):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.camera = None
        self.camera_resolution = camera_resolution
        self.detection_history = deque(maxlen=10)
        self.detection_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.camera_thread = None
        self.camera_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.target_classes = {'person', 'cat', 'dog', 'chair', 'potted plant', 'bicycle'}

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"✅ YOLO 모델 '{model_path}' 로드 완료")
            except Exception as e:
                print(f"❌ YOLO 모델 로드 실패: {e}")
        else:
            print("❌ YOLO 라이브러리가 설치되지 않아 검출 기능을 사용할 수 없습니다.")

        if PI_CAMERA_AVAILABLE:
            self.initialize_pi_camera()

    def initialize_pi_camera(self):
        try:
            if USE_LEGACY_PICAMERA:
                self.camera = picamera.PiCamera()
                self.camera.resolution = self.camera_resolution
                self.camera.framerate = 24
                time.sleep(2)
                print("✅ Pi 카메라 (Legacy) 초기화 완료")
            else:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(main={"size": self.camera_resolution})
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)
                print("✅ Pi 카메라 (Picamera2) 초기화 완료")
        except Exception as e:
            print(f"❌ Pi 카메라 초기화 실패: {e}")
            self.camera = None

    def start_camera_thread(self):
        if not self.camera:
            print("❌ 카메라가 없어 스트리밍 스레드를 시작할 수 없습니다.")
            return
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self.camera_thread.start()
        print("📹 Pi 카메라 스트리밍 시작")

    def stop_camera_thread(self):
        self.camera_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        print("📹 Pi 카메라 스트리밍 중지")

    def _camera_capture_loop(self):
        while self.camera_running:
            try:
                frame = self._capture_single_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                time.sleep(0.1) # approx 10 FPS
            except Exception as e:
                print(f"카메라 캡처 루프 오류: {e}")
                time.sleep(1)

    def _capture_single_frame(self) -> Optional[np.ndarray]:
        if not self.camera: return None
        with self.camera_lock:
            if USE_LEGACY_PICAMERA:
                with picamera.array.PiRGBArray(self.camera) as stream:
                    self.camera.capture(stream, format='bgr')
                    return stream.array
            else:
                frame = self.camera.capture_array()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
        if not self.model: return []
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        detections = []
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    if class_name in self.target_classes:
                        detections.append(DetectionResult(
                            class_name=class_name,
                            confidence=float(box.conf[0]),
                            bbox=box.xyxy[0].tolist(),
                            timestamp=time.time()
                        ))
        with self.detection_lock:
            self.detection_history.append(detections)
        return detections

    def has_obstacle_in_direction(self, direction: str = "forward") -> Tuple[bool, List[str]]:
        frame = self.get_latest_frame()
        if frame is None: return False, []
        
        detections = self.detect_objects(frame)
        detected_dangers = []
        img_width = self.camera_resolution[0]
        
        for detection in detections:
            bbox_center_x = (detection.bbox[0] + detection.bbox[2]) / 2
            is_in_danger_zone = False
            if direction == "forward" and (img_width * 0.25 <= bbox_center_x <= img_width * 0.75):
                is_in_danger_zone = True
            
            if is_in_danger_zone and detection.class_name not in detected_dangers:
                detected_dangers.append(detection.class_name)
        
        return len(detected_dangers) > 0, detected_dangers

    def cleanup(self):
        self.stop_camera_thread()
        if self.camera:
            try:
                if USE_LEGACY_PICAMERA:
                    self.camera.close()
                else:
                    self.camera.stop()
            except Exception as e:
                print(f"카메라 정리 중 오류: {e}")
        print("🧹 YOLO 검출기 정리 완료")

# ----------------------------------------------------------------------------------
# 🔄 ENHANCED: YOLO 통합 경로 계획 클래스
# ----------------------------------------------------------------------------------
class IntelligentPathPlanner:
    def __init__(self, room_database, ollama_url="http://localhost:11434/api/chat", model_name="llama3:latest"):
        self.room_database = room_database
        self.ollama_url = ollama_url
        self.model_name = model_name

    def _calculate_path_distance(self, waypoints: list) -> float:
        total_distance = 0
        for i in range(len(waypoints) - 1):
            start_coords = self.room_database[waypoints[i]]
            end_coords = self.room_database[waypoints[i+1]]
            total_distance += np.sqrt((start_coords[0] - end_coords[0])**2 + (start_coords[1] - end_coords[1])**2)
        return round(total_distance, 2)

    def generate_path_candidates(self, start_room: str, end_room: str, max_paths: int = 5, excluded_waypoints: List[str] = None) -> list:
        room_list = list(self.room_database.keys())
        if start_room not in room_list or end_room not in room_list: return []
        
        intermediate_rooms = [r for r in room_list if r not in [start_room, end_room]]
        if excluded_waypoints:
             intermediate_rooms = [r for r in intermediate_rooms if r not in excluded_waypoints]

        path_candidates = []
        
        # 1. 직선 경로
        direct_path = [start_room, end_room]
        path_candidates.append({"path_id": 1, "waypoints": direct_path, "description": "가장 짧은 직접 경로", "estimated_distance": self._calculate_path_distance(direct_path)})
        
        # 2. 우회 경로
        for i, intermediate in enumerate(intermediate_rooms):
            if len(path_candidates) >= max_paths: break
            waypoints = [start_room, intermediate, end_room]
            path_candidates.append({"path_id": len(path_candidates) + 1, "waypoints": waypoints, "description": f"'{intermediate}' 방을 경유하는 우회 경로", "estimated_distance": self._calculate_path_distance(waypoints)})
        
        return path_candidates

    def evaluate_and_select_path(self, path_candidates: list, full_command: str, rulebook: list, current_path_to_avoid: Optional[List[str]] = None, detected_obstacles: Optional[List[str]] = None):
        print("\n   - LLM이 생성된 경로 후보들을 규칙과 실시간 상황에 따라 평가합니다...")

        rulebook_section = "현재 영구 규칙서(Rulebook)에 등록된 규칙이 없습니다."
        if rulebook:
            rules_str = "\n".join([f"  - {r}" for r in rulebook])
            rulebook_section = f"다음은 반드시 지켜야 하는 영구 규칙입니다:\n{rules_str}"

        yolo_section = ""
        if detected_obstacles:
            yolo_section = f"🚨 **실시간 시각 정보**: 현재 로봇 전방에 '{', '.join(detected_obstacles)}' (이)가 감지되었습니다. 안전을 위해 이 상황을 반드시 고려해야 합니다."

        avoidance_section = ""
        if current_path_to_avoid:
            avoidance_section = f"⚠️ **경로 재계획 요청**: 현재 경로({' → '.join(current_path_to_avoid)})에서 장애물이 지속적으로 감지되어 다른 경로를 찾아야 합니다. 이 경로는 선택하지 마세요."

        prompt = f"""
당신은 로봇의 경로를 결정하는 지능적인 네비게이션 전문가입니다. 여러 경로 후보 중 가장 안전하고 효율적인 최적의 경로를 단 하나만 선택해야 합니다.

**상황 분석:**
1.  **전체 임무**: "{full_command}"
2.  **영구 규칙 (Rulebook)**: {rulebook_section}
3.  **일시적 명령**: 전체 임무 내용에서 '피해서', '거치지 말고' 등과 같은 일시적인 제약 조건이 있는지 확인하세요.
4.  {yolo_section}
5.  {avoidance_section}

**평가할 경로 후보 목록:**
{json.dumps(path_candidates, indent=2, ensure_ascii=False)}

**결정 프로세스:**
1.  **안전 최우선**: `경로 재계획 요청`이나 `실시간 시각 정보`가 있다면, 이를 위반하는 경로는 최우선으로 제외하세요.
2.  **규칙 준수**: 영구 규칙과 일시적 명령을 위반하는 경로는 그 다음으로 탈락시키세요.
3.  **효율성**: 모든 안전 및 규칙 조건을 만족하는 경로들 중에서, 가장 이동 거리가 짧은 경로를 선택하세요.

**출력:**
선택한 경로의 `path_id` 숫자 하나만 응답하세요. 다른 설명은 절대 추가하지 마세요.
"""
        try:
            payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "stream": False}
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '1').strip()
            numbers = re.findall(r'\d+', content)
            selected_id = int(numbers[0]) if numbers else 1
            
            selected_path = next((p for p in path_candidates if p["path_id"] == selected_id), path_candidates[0])
            print(f"   - LLM의 선택: 경로 {selected_path['path_id']} ({' → '.join(selected_path['waypoints'])})")
            return selected_path

        except Exception as e:
            print(f"   - LLM 경로 평가 실패: {e}. 안전을 위해 가장 짧은 경로를 선택합니다.")
            return path_candidates[0]
            
# ----------------------------------------------------------------------------------
# 🚀 UPGRADED: 행동 계획 및 실행 시스템
# ----------------------------------------------------------------------------------
class CommandExecutor:
    def __init__(self, room_database, yolo_detector: YOLODetector):
        self.room_database = room_database
        self.yolo_detector = yolo_detector
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model_name = "llama3:latest"
        self.navigator = BasicNavigator()
        self.rulebook = []
        self.current_location = 'A' # 초기 위치 가정
        self.path_planner = IntelligentPathPlanner(self.room_database, self.ollama_url, self.model_name)
        
        self.navigator.waitUntilNav2Active()
        print("Nav2 시스템 활성화 완료. 5초 후 초기 위치를 설정합니다...")
        time.sleep(5)
        
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = -1.867; initial_pose.pose.position.y = -0.435
        q = tf_transformations.quaternion_from_euler(0, 0, -0.015)
        initial_pose.pose.orientation.x, initial_pose.pose.orientation.y, initial_pose.pose.orientation.z, initial_pose.pose.orientation.w = q
        self.navigator.setInitialPose(initial_pose)
        self.update_initial_location()
        print("초기 위치 설정 완료.")

    def add_rule(self, rule_text: str):
        self.rulebook.append(rule_text)
        print(f"✅ 새로운 규칙 추가: '{rule_text}'")

    def update_initial_location(self):
        spawn_point = np.array([-1.867, -0.435])
        closest_room = min(self.room_database.keys(), key=lambda room: np.linalg.norm(spawn_point - np.array(self.room_database[room])))
        self.current_location = closest_room
        print(f"로봇의 실제 시작 위치를 '{closest_room}' 방으로 자동 설정했습니다.")

    def parse_command_to_high_level_plan(self, command: str) -> list:
        print(f"\n[분석] '{command}' 명령을 분석하여 상위 계획을 수립합니다...")
        prompt = f"""
You are a command interpreter for a mobile robot. Convert the user's command into a structured JSON plan.
The robot knows these rooms: {list(self.room_database.keys())}.
The user's command is: "{command}"
Extract the sequence of destinations.
Respond ONLY with a JSON object in the format: {{"plan": [{{"destination": "A"}}, {{"destination": "B"}}]}}
"""
        try:
            payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "stream": False, "format": "json"}
            response = requests.post(self.ollama_url, json=payload, timeout=180)
            response.raise_for_status()
            plan = json.loads(response.json().get('message', {}).get('content', '{}')).get("plan", [])
            print("  - 상위 계획 수립 완료:", ' -> '.join([p['destination'] for p in plan]))
            return plan
        except Exception as e:
            print(f"  - LLM 상위 계획 수립 실패: {e}"); return []

    def execute_multi_step_plan(self, high_level_plan: list, original_command: str):
        if not high_level_plan:
            print("실행할 계획이 없습니다."); return
            
        print("\n[임무 시작] 최종 계획을 바탕으로 임무를 시작합니다.")
        
        for i, step in enumerate(high_level_plan):
            destination = step.get("destination")
            if not destination or destination == self.current_location:
                print(f"이미 '{destination}'에 있거나 유효하지 않은 목적지입니다. 다음 단계로 넘어갑니다.")
                continue

            print(f"\n--- [세부 임무 {i+1}/{len(high_level_plan)}] '{self.current_location}'에서 '{destination}'(으)로 이동 ---")
            
            # 1. 경로 후보 생성 및 LLM을 통한 최적 경로 선택
            candidates = self.path_planner.generate_path_candidates(self.current_location, destination)
            if not candidates:
                print("   - 경로 후보 생성 실패. 이 임무를 중단합니다."); break
            
            best_path = self.path_planner.evaluate_and_select_path(candidates, original_command, self.rulebook)
            
            # 2. 선택된 경로(웨이포인트)를 따라 이동 (실시간 장애물 감지 포함)
            path_succeeded = self.execute_path_with_obstacle_avoidance(best_path, candidates, original_command)
            
            if not path_succeeded:
                print(f"'{destination}'(으)로 이동하는 데 최종 실패했습니다. 전체 임무를 중단합니다."); break
        
        print("\n🎉 모든 임무를 완료했습니다!")

    def execute_path_with_obstacle_avoidance(self, current_path: dict, candidates: list, original_command: str) -> bool:
        """경로를 실행하되, 장애물 발견 시 재탐색 로직을 포함"""
        waypoints_to_execute = current_path['waypoints']
        
        for i, destination_label in enumerate(waypoints_to_execute[1:]):
            if destination_label not in self.room_database:
                print(f"    - '{destination_label}'은(는) 알 수 없는 방입니다. 건너뜁니다."); continue
            
            coords = self.room_database[destination_label]
            print(f"    - [실행] 웨이포인트: '{self.current_location}' → '{destination_label}' 이동 시작...")

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x, goal_pose.pose.position.y = coords[0], coords[1]
            goal_pose.pose.orientation.w = 1.0
            
            self.navigator.goToPose(goal_pose)
            
            obstacle_count = 0
            replan_threshold = 3 # 3번 연속 감지되면 재계획
            
            while not self.navigator.isTaskComplete():
                has_obstacle, detected_objects = self.yolo_detector.has_obstacle_in_direction("forward")
                if has_obstacle:
                    obstacle_count += 1
                    print(f"    ⚠️ 전방 장애물 감지됨: {detected_objects} (카운트: {obstacle_count}/{replan_threshold})")
                    if obstacle_count >= replan_threshold:
                        print("    - [조치] 장애물 지속 감지. 현재 경로를 취소하고 재탐색합니다.")
                        self.navigator.cancelTask()
                        time.sleep(1.0) # 취소 시간 확보
                        
                        # 현재 경로를 제외한 새로운 경로 탐색
                        remaining_candidates = [p for p in candidates if p['path_id'] != current_path['path_id']]
                        if not remaining_candidates:
                            print("    - [실패] 더 이상 시도할 대안 경로가 없습니다.")
                            return False
                        
                        print("    - [재탐색] 대안 경로를 LLM으로 재평가합니다.")
                        new_path = self.path_planner.evaluate_and_select_path(
                            remaining_candidates, 
                            original_command, 
                            self.rulebook, 
                            current_path_to_avoid=current_path['waypoints'],
                            detected_obstacles=detected_objects
                        )
                        
                        # 재귀적으로 새로운 경로 실행
                        return self.execute_path_with_obstacle_avoidance(new_path, remaining_candidates, original_command)
                else:
                    obstacle_count = 0 # 장애물이 없으면 카운트 초기화
                
                time.sleep(0.5) # 2Hz로 장애물 체크

            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                print(f"    - '{destination_label}' 도착 성공!")
                self.current_location = destination_label
            elif result == TaskResult.CANCELED:
                # 재탐색으로 인한 취소는 실패가 아님
                print(f"    - '{destination_label}'로의 이동이 재탐색을 위해 취소되었습니다.")
                return False # 재귀 호출에서 처리했으므로 여기선 실패로 간주
            else:
                print(f"    - '{destination_label}' 도착 실패 (결과: {result}).")
                return False
        return True

# ----------------------------------------------------------------------------------
# 메인 실행 로직
# ----------------------------------------------------------------------------------
def main():
    rclpy.init()
    
    merger = IntelligentRoomMerger()
    yaml_file_path = "~/maps/house.yaml"
    room_db = merger.get_room_database(yaml_file_path)

    if not room_db:
        print("방 데이터베이스를 생성하지 못했습니다. 프로그램을 종료합니다.")
        rclpy.shutdown(); return
    
    # YOLO 검출기 초기화 및 카메라 스레드 시작
    yolo_detector = YOLODetector()
    yolo_detector.start_camera_thread()

    executor = CommandExecutor(room_db, yolo_detector)
    
    try:
        while True:
            command = input("\n명령을 입력하세요 (예: A 갔다가 C 피해서 D 가줘 / 규칙추가: C에서는 천천히 / exit): ")
            if command.lower() == 'exit': break
            
            if command.lower().startswith('규칙추가:'):
                rule = command.split(':', 1)[1].strip()
                if rule: executor.add_rule(rule)
                else: print("오류: 추가할 규칙 내용이 없습니다.")
                continue
            
            high_level_plan = executor.parse_command_to_high_level_plan(command)
            
            if high_level_plan:
                executor.execute_multi_step_plan(high_level_plan, command)
            else:
                print("유효한 계획이 수립되지 않았습니다.")

    except KeyboardInterrupt:
        print("\n프로그램을 강제 종료합니다.")
    finally:
        print("시스템을 종료합니다...")
        yolo_detector.cleanup() # YOLO 및 카메라 리소스 정리
        executor.navigator.lifecycleShutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
