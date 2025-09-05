#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import numpy as np

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
import geometry_msgs.msg

from llm_path_planning.ollama_client import OllamaClient
from llm_path_planning.map_processor import MapProcessor
from llm_path_planning.path_evaluator import PathEvaluator


class LLMPathPlanner(Node):
    """LLM 기반 경로 계획 노드"""
    
    def __init__(self):
        super().__init__('llm_path_planner')
        
        # 파라미터 선언
        self.declare_parameters(
            namespace='',
            parameters=[
                ('ollama_host', 'localhost'),
                ('ollama_port', 11434),
                ('ollama_model', 'llama3.1:8b'),
                ('num_candidates', 5),
                ('robot_radius', 0.2),
                ('safety_margin', 0.1),
                ('downsample_factor', 4),
                ('occupied_threshold', 0.65),
                ('free_threshold', 0.25),
                ('weights.collision_penalty', 1000.0),
                ('weights.path_length', 1.0),
                ('weights.smoothness', 0.5),
                ('weights.clearance', 2.0),
                ('topics.map_topic', '/map'),
                ('topics.amcl_pose_topic', '/amcl_pose'),
                ('topics.goal_pose_topic', '/goal_pose'),
                ('topics.candidate_paths_topic', '/candidate_paths'),
                ('topics.selected_path_topic', '/selected_path'),
            ]
        )
        
        # 파라미터 읽기
        self.ollama_host = self.get_parameter('ollama_host').get_parameter_value().string_value
        self.ollama_port = self.get_parameter('ollama_port').get_parameter_value().integer_value
        self.ollama_model = self.get_parameter('ollama_model').get_parameter_value().string_value
        self.num_candidates = self.get_parameter('num_candidates').get_parameter_value().integer_value
        self.robot_radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value
        self.downsample_factor = self.get_parameter('downsample_factor').get_parameter_value().integer_value
        self.occupied_threshold = self.get_parameter('occupied_threshold').get_parameter_value().double_value
        self.free_threshold = self.get_parameter('free_threshold').get_parameter_value().double_value
        
        # 가중치
        self.weights = {
            'collision_penalty': self.get_parameter('weights.collision_penalty').get_parameter_value().double_value,
            'path_length': self.get_parameter('weights.path_length').get_parameter_value().double_value,
            'smoothness': self.get_parameter('weights.smoothness').get_parameter_value().double_value,
            'clearance': self.get_parameter('weights.clearance').get_parameter_value().double_value,
        }
        
        # 토픽 이름
        self.map_topic = self.get_parameter('topics.map_topic').get_parameter_value().string_value
        self.amcl_pose_topic = self.get_parameter('topics.amcl_pose_topic').get_parameter_value().string_value
        self.goal_pose_topic = self.get_parameter('topics.goal_pose_topic').get_parameter_value().string_value
        self.candidate_paths_topic = self.get_parameter('topics.candidate_paths_topic').get_parameter_value().string_value
        self.selected_path_topic = self.get_parameter('topics.selected_path_topic').get_parameter_value().string_value
        
        # 컴포넌트 초기화
        self.ollama_client = OllamaClient(self.ollama_host, self.ollama_port, self.ollama_model)
        self.map_processor = MapProcessor(self.downsample_factor, self.occupied_threshold, self.free_threshold)
        self.path_evaluator = PathEvaluator(self.map_processor, self.weights, self.robot_radius, self.safety_margin)
        
        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 구독자
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, qos_profile)
        self.amcl_pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, self.amcl_pose_topic, self.amcl_pose_callback, qos_profile)
        self.goal_pose_subscriber = self.create_subscription(
            PoseStamped, self.goal_pose_topic, self.goal_pose_callback, qos_profile)
        
        # 발행자
        self.candidate_paths_publisher = self.create_publisher(
            MarkerArray, self.candidate_paths_topic, qos_profile)
        self.selected_path_publisher = self.create_publisher(
            Path, self.selected_path_topic, qos_profile)
        
        # 상태 변수
        self.current_map = None
        self.current_map_data = None
        self.current_pose = None
        self.current_goal = None
        
        # 시각화 색상
        self.candidate_colors = [
            [1.0, 0.0, 0.0, 0.7],  # Red
            [0.0, 1.0, 0.0, 0.7],  # Green
            [0.0, 0.0, 1.0, 0.7],  # Blue
            [1.0, 1.0, 0.0, 0.7],  # Yellow
            [1.0, 0.0, 1.0, 0.7],  # Magenta
        ]
        
        self.get_logger().info('LLM Path Planner 노드가 시작되었습니다.')
        self.get_logger().info(f'Ollama 서버: {self.ollama_host}:{self.ollama_port}')
        self.get_logger().info(f'사용 모델: {self.ollama_model}')
    
    def map_callback(self, msg):
        """맵 정보 업데이트"""
        self.current_map = msg
        self.current_map_data = self.map_processor.process_map(msg)
        self.get_logger().info(f'맵 업데이트: {msg.info.width}x{msg.info.height} -> {self.current_map_data["width"]}x{self.current_map_data["height"]}')
        
        # 모든 정보가 준비되면 경로 계획 실행
        self.try_plan_path()
    
    def amcl_pose_callback(self, msg):
        """현재 로봇 위치 업데이트"""
        self.current_pose = msg.pose.pose
        self.get_logger().debug(f'로봇 위치 업데이트: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})')
        
        # 모든 정보가 준비되면 경로 계획 실행
        self.try_plan_path()
    
    def goal_pose_callback(self, msg):
        """목표 위치 업데이트"""
        self.current_goal = msg.pose
        self.get_logger().info(f'목표 위치 업데이트: ({self.current_goal.position.x:.2f}, {self.current_goal.position.y:.2f})')
        
        # 모든 정보가 준비되면 경로 계획 실행
        self.try_plan_path()
    
    def try_plan_path(self):
        """모든 정보가 준비되면 경로 계획 실행"""
        if (self.current_map_data is None or 
            self.current_pose is None or 
            self.current_goal is None):
            return
        
        self.get_logger().info('경로 계획을 시작합니다...')
        self.plan_path()
    
    def plan_path(self):
        """LLM을 사용한 경로 계획 실행"""
        try:
            # 시작점과 목표점을 그리드 좌표로 변환
            start_grid = self.map_processor.world_to_grid(
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_map_data
            )
            
            goal_grid = self.map_processor.world_to_grid(
                self.current_goal.position.x,
                self.current_goal.position.y,
                self.current_map_data
            )
            
            self.get_logger().info(f'시작점 (그리드): {start_grid}')
            self.get_logger().info(f'목표점 (그리드): {goal_grid}')
            
            # LLM으로부터 후보 경로 생성
            start_list = [float(start_grid[0]), float(start_grid[1])]
            goal_list = [float(goal_grid[0]), float(goal_grid[1])]
            
            self.get_logger().info('LLM에서 경로 후보를 생성중...')
            response = self.ollama_client.generate_path_candidates(
                self.current_map_data, start_list, goal_list, self.num_candidates)
            
            candidates = response.get('candidates', [])
            self.get_logger().info(f'LLM에서 {len(candidates)}개의 후보 경로를 생성했습니다.')
            
            if not candidates:
                self.get_logger().warn('생성된 후보 경로가 없습니다.')
                return
            
            # 후보 경로 검증 및 점수화
            self.get_logger().info('후보 경로를 평가중...')
            for i, candidate in enumerate(candidates):
                # 웨이포인트 검증
                waypoints = candidate.get('waypoints', [])
                validated_waypoints = self.path_evaluator.validate_waypoints(waypoints, self.current_map_data)
                candidate['waypoints'] = validated_waypoints
            
            scored_candidates = self.path_evaluator.evaluate_candidates(candidates, self.current_map_data)
            
            # 최적 후보 선택
            best_candidate = self.path_evaluator.get_best_candidate(scored_candidates)
            
            if best_candidate is None:
                self.get_logger().warn('유효한 경로 후보를 찾을 수 없습니다.')
                return
            
            self.get_logger().info(f'최적 경로 선택: {best_candidate["id"]} (점수: {best_candidate["total_score"]:.2f})')
            
            # 결과 시각화
            self.visualize_candidate_paths(scored_candidates)
            self.publish_selected_path(best_candidate)
            
            # 결과 로그
            for candidate in scored_candidates:
                self.get_logger().info(
                    f'후보 {candidate["id"]}: 점수={candidate["total_score"]:.2f}, '
                    f'충돌={candidate["collision_score"]:.2f}, '
                    f'길이={candidate["length_score"]:.2f}, '
                    f'유효={candidate["is_valid"]}'
                )
                
        except Exception as e:
            self.get_logger().error(f'경로 계획 중 오류 발생: {e}')
    
    def visualize_candidate_paths(self, candidates):
        """후보 경로들을 MarkerArray로 시각화"""
        marker_array = MarkerArray()
        
        for i, candidate in enumerate(candidates):
            waypoints = candidate.get('waypoints', [])
            if len(waypoints) < 2:
                continue
            
            marker = Marker()
            marker.header.frame_id = self.current_map.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "candidate_paths"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # 라인 두께
            
            # 색상 설정
            color_idx = i % len(self.candidate_colors)
            marker.color.r = self.candidate_colors[color_idx][0]
            marker.color.g = self.candidate_colors[color_idx][1]
            marker.color.b = self.candidate_colors[color_idx][2]
            marker.color.a = self.candidate_colors[color_idx][3]
            
            # 웨이포인트를 월드 좌표로 변환하여 추가
            for waypoint in waypoints:
                world_pos = self.map_processor.grid_to_world(
                    int(waypoint[0]), int(waypoint[1]), self.current_map_data)
                
                point = Point()
                point.x = world_pos[0]
                point.y = world_pos[1]
                point.z = 0.1
                marker.points.append(point)
            
            marker_array.markers.append(marker)
        
        self.candidate_paths_publisher.publish(marker_array)
        self.get_logger().info(f'{len(marker_array.markers)}개의 후보 경로를 시각화했습니다.')
    
    def publish_selected_path(self, best_candidate):
        """선택된 최적 경로를 Path 메시지로 발행"""
        path_msg = Path()
        path_msg.header.frame_id = self.current_map.header.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        waypoints = best_candidate.get('waypoints', [])
        
        for waypoint in waypoints:
            world_pos = self.map_processor.grid_to_world(
                int(waypoint[0]), int(waypoint[1]), self.current_map_data)
            
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = world_pos[0]
            pose_stamped.pose.position.y = world_pos[1]
            pose_stamped.pose.position.z = 0.0
            
            # 간단한 방향 계산 (다음 웨이포인트 방향)
            pose_stamped.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose_stamped)
        
        self.selected_path_publisher.publish(path_msg)
        self.get_logger().info(f'선택된 경로를 발행했습니다. (웨이포인트 {len(waypoints)}개)')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LLMPathPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'노드 실행 중 오류: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
