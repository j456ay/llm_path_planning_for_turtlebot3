#!/usr/bin/env python3

import numpy as np
import math
from typing import Dict, List, Tuple, Any
from .map_processor import MapProcessor


class PathEvaluator:
    """경로 후보들을 평가하고 점수화"""
    
    def __init__(self, map_processor: MapProcessor, weights: Dict[str, float], 
                 robot_radius: float = 0.2, safety_margin: float = 0.1):
        self.map_processor = map_processor
        self.weights = weights
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        
    def evaluate_candidates(self, candidates: List[Dict[str, Any]], 
                          map_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """후보 경로들을 평가하고 점수화"""
        scored_candidates = []
        
        for candidate in candidates:
            waypoints = candidate.get('waypoints', [])
            
            if len(waypoints) < 2:
                continue
                
            # 각 평가 지표 계산
            collision_score = self._evaluate_collision(waypoints, map_data)
            length_score = self._evaluate_path_length(waypoints, map_data)
            smoothness_score = self._evaluate_smoothness(waypoints)
            clearance_score = self._evaluate_clearance(waypoints, map_data)
            
            # 가중합으로 최종 점수 계산 (낮을수록 좋음)
            total_score = (
                collision_score * self.weights.get('collision_penalty', 1000.0) +
                length_score * self.weights.get('path_length', 1.0) +
                smoothness_score * self.weights.get('smoothness', 0.5) +
                clearance_score * self.weights.get('clearance', 2.0)
            )
            
            scored_candidate = candidate.copy()
            scored_candidate.update({
                'total_score': total_score,
                'collision_score': collision_score,
                'length_score': length_score,
                'smoothness_score': smoothness_score,
                'clearance_score': clearance_score,
                'is_valid': collision_score == 0  # 충돌이 없으면 유효
            })
            
            scored_candidates.append(scored_candidate)
        
        # 점수로 정렬 (낮은 점수가 더 좋음)
        scored_candidates.sort(key=lambda x: x['total_score'])
        
        return scored_candidates
    
    def _evaluate_collision(self, waypoints: List[List[float]], map_data: Dict[str, Any]) -> float:
        """충돌 검사 - 충돌이 있으면 큰 페널티"""
        if len(waypoints) < 2:
            return 1.0
        
        total_collision = 0.0
        
        # 각 웨이포인트 사이의 경로를 체크
        for i in range(len(waypoints) - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]
            
            # 직선 경로를 따라 충돌 체크
            collision_count = self._check_line_collision(start_point, end_point, map_data)
            total_collision += collision_count
        
        return total_collision
    
    def _check_line_collision(self, start: List[float], end: List[float], 
                            map_data: Dict[str, Any]) -> float:
        """두 점 사이의 직선 경로에서 충돌 체크"""
        start_x, start_y = int(round(start[0])), int(round(start[1]))
        end_x, end_y = int(round(end[0])), int(round(end[1]))
        
        # Bresenham's line algorithm으로 직선상의 모든 점 체크
        points = self._get_line_points(start_x, start_y, end_x, end_y)
        
        collision_count = 0
        for point in points:
            x, y = point
            if not self.map_processor.is_cell_free(x, y, map_data, 
                                                 self.robot_radius + self.safety_margin):
                collision_count += 1
        
        return collision_count / max(len(points), 1)
    
    def _get_line_points(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm으로 직선상의 모든 점 반환"""
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
                
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def _evaluate_path_length(self, waypoints: List[List[float]], map_data: Dict[str, Any]) -> float:
        """경로 길이 평가"""
        if len(waypoints) < 2:
            return float('inf')
        
        total_length = 0.0
        
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            
            # 그리드 좌표를 월드 좌표로 변환하여 실제 거리 계산
            world_p1 = self.map_processor.grid_to_world(int(p1[0]), int(p1[1]), map_data)
            world_p2 = self.map_processor.grid_to_world(int(p2[0]), int(p2[1]), map_data)
            
            distance = math.sqrt((world_p2[0] - world_p1[0])**2 + (world_p2[1] - world_p1[1])**2)
            total_length += distance
        
        return total_length
    
    def _evaluate_smoothness(self, waypoints: List[List[float]]) -> float:
        """경로 부드러움 평가 - 급격한 방향 변화에 페널티"""
        if len(waypoints) < 3:
            return 0.0
        
        total_curvature = 0.0
        
        for i in range(1, len(waypoints) - 1):
            p1 = waypoints[i - 1]
            p2 = waypoints[i]
            p3 = waypoints[i + 1]
            
            # 벡터 계산
            v1 = [p2[0] - p1[0], p2[1] - p1[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]
            
            # 벡터 크기
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 > 0 and len2 > 0:
                # 코사인 각도
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
                cos_angle = max(-1, min(1, cos_angle))  # 범위 제한
                
                # 각도 변화량 (0~π)
                angle_change = math.acos(cos_angle)
                total_curvature += angle_change
        
        return total_curvature / max(len(waypoints) - 2, 1)
    
    def _evaluate_clearance(self, waypoints: List[List[float]], map_data: Dict[str, Any]) -> float:
        """장애물 클리어런스 평가 - 장애물에 가까우면 페널티"""
        if len(waypoints) == 0:
            return float('inf')
        
        total_clearance_penalty = 0.0
        
        for waypoint in waypoints:
            x, y = int(round(waypoint[0])), int(round(waypoint[1]))
            clearance = self.map_processor.get_clearance(x, y, map_data)
            
            # 클리어런스가 작을수록 큰 페널티 (역수 관계)
            if clearance > 0:
                penalty = 1.0 / (clearance + 0.1)  # 0.1을 더해서 무한대 방지
            else:
                penalty = 10.0  # 매우 큰 페널티
            
            total_clearance_penalty += penalty
        
        return total_clearance_penalty / len(waypoints)
    
    def get_best_candidate(self, scored_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """가장 좋은 후보 반환"""
        if not scored_candidates:
            return None
        
        # 유효한 후보들 중에서 선택
        valid_candidates = [c for c in scored_candidates if c.get('is_valid', False)]
        
        if valid_candidates:
            return valid_candidates[0]  # 이미 점수순으로 정렬되어 있음
        else:
            # 유효한 후보가 없으면 충돌 점수가 가장 낮은 것 선택
            return min(scored_candidates, key=lambda x: x.get('collision_score', float('inf')))
    
    def validate_waypoints(self, waypoints: List[List[float]], map_data: Dict[str, Any]) -> List[List[float]]:
        """웨이포인트들을 검증하고 필요시 수정"""
        if not waypoints:
            return waypoints
        
        validated_waypoints = []
        
        for waypoint in waypoints:
            x, y = int(round(waypoint[0])), int(round(waypoint[1]))
            
            # 맵 경계 내로 제한
            x = max(0, min(x, map_data['width'] - 1))
            y = max(0, min(y, map_data['height'] - 1))
            
            # 만약 해당 위치가 충돌이면 주변에서 안전한 위치 찾기
            if not self.map_processor.is_cell_free(x, y, map_data, self.robot_radius):
                safe_point = self._find_nearest_safe_point(x, y, map_data)
                if safe_point:
                    x, y = safe_point
            
            validated_waypoints.append([float(x), float(y)])
        
        return validated_waypoints
    
    def _find_nearest_safe_point(self, x: int, y: int, map_data: Dict[str, Any], 
                               max_search_radius: int = 5) -> Tuple[int, int]:
        """가장 가까운 안전한 지점 찾기"""
        for radius in range(1, max_search_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy > radius*radius:
                        continue
                    
                    check_x = x + dx
                    check_y = y + dy
                    
                    if (check_x >= 0 and check_x < map_data['width'] and 
                        check_y >= 0 and check_y < map_data['height']):
                        
                        if self.map_processor.is_cell_free(check_x, check_y, map_data, self.robot_radius):
                            return check_x, check_y
        
        return None
