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
    
    def _get_line_
