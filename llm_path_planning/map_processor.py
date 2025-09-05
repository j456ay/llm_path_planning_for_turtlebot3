#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Tuple, Any
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point


class MapProcessor:
    """OccupancyGrid 맵 처리 및 저해상도 변환"""
    
    def __init__(self, downsample_factor: int = 4, occupied_threshold: float = 0.65, 
                 free_threshold: float = 0.25):
        self.downsample_factor = downsample_factor
        self.occupied_threshold = occupied_threshold
        self.free_threshold = free_threshold
        
    def process_map(self, occupancy_grid: OccupancyGrid) -> Dict[str, Any]:
        """OccupancyGrid를 처리하여 LLM 입력용 데이터로 변환"""
        
        # 원본 맵 정보
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        resolution = occupancy_grid.info.resolution
        origin_x = occupancy_grid.info.origin.position.x
        origin_y = occupancy_grid.info.origin.position.y
        
        # 맵 데이터를 2D numpy 배열로 변환
        map_data = np.array(occupancy_grid.data, dtype=np.int8).reshape((height, width))
        
        # 다운샘플링
        downsampled_map = self._downsample_map(map_data)
        
        # 블록된 셀 찾기
        blocked_cells = self._find_blocked_cells(downsampled_map)
        
        # 새로운 맵 정보
        new_width = downsampled_map.shape[1]
        new_height = downsampled_map.shape[0]
        new_resolution = resolution * self.downsample_factor
        
        return {
            'width': new_width,
            'height': new_height,
            'resolution': new_resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'downsample_factor': self.downsample_factor,
            'blocked_cells': blocked_cells,
            'map_array': downsampled_map,
            'original_resolution': resolution,
            'original_width': width,
            'original_height': height
        }
    
    def _downsample_map(self, map_data: np.ndarray) -> np.ndarray:
        """맵을 다운샘플링"""
        height, width = map_data.shape
        factor = self.downsample_factor
        
        # 새로운 크기 계산
        new_height = height // factor
        new_width = width // factor
        
        # 다운샘플링된 맵 초기화
        downsampled = np.zeros((new_height, new_width), dtype=np.int8)
        
        for i in range(new_height):
            for j in range(new_width):
                # factor x factor 영역의 대표값 계산
                y_start = i * factor
                y_end = min((i + 1) * factor, height)
                x_start = j * factor
                x_end = min((j + 1) * factor, width)
                
                region = map_data[y_start:y_end, x_start:x_end]
                
                # unknown(-1)이 있으면 unknown으로
                if np.any(region == -1):
                    downsampled[i, j] = -1
                # occupied(100) 비율이 임계값 이상이면 occupied
                elif np.mean(region >= self.occupied_threshold * 100) > 0.3:
                    downsampled[i, j] = 100
                # free(0) 비율이 임계값 이상이면 free  
                elif np.mean(region <= self.free_threshold * 100) > 0.7:
                    downsampled[i, j] = 0
                else:
                    downsampled[i, j] = -1  # unknown
                    
        return downsampled
    
    def _find_blocked_cells(self, map_data: np.ndarray) -> List[List[int]]:
        """블록된 셀들의 좌표 리스트 반환"""
        blocked_cells = []
        height, width = map_data.shape
        
        for i in range(height):
            for j in range(width):
                if map_data[i, j] >= self.occupied_threshold * 100:
                    blocked_cells.append([j, height - 1 - i])  # ROS 좌표계로 변환
                    
        return blocked_cells
    
    def world_to_grid(self, world_x: float, world_y: float, map_data: Dict[str, Any]) -> Tuple[int, int]:
        """월드 좌표를 다운샘플링된 그리드 좌표로 변환"""
        # 원본 그리드 좌표로 변환
        orig_grid_x = int((world_x - map_data['origin_x']) / map_data['original_resolution'])
        orig_grid_y = int((world_y - map_data['origin_y']) / map_data['original_resolution'])
        
        # 다운샘플링된 그리드 좌표로 변환
        grid_x = orig_grid_x // map_data['downsample_factor']
        grid_y = orig_grid_y // map_data['downsample_factor']
        
        # 범위 체크
        grid_x = max(0, min(grid_x, map_data['width'] - 1))
        grid_y = max(0, min(grid_y, map_data['height'] - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int, map_data: Dict[str, Any]) -> Tuple[float, float]:
        """다운샘플링된 그리드 좌표를 월드 좌표로 변환"""
        world_x = map_data['origin_x'] + (grid_x * map_data['downsample_factor'] + 0.5) * map_data['original_resolution']
        world_y = map_data['origin_y'] + (grid_y * map_data['downsample_factor'] + 0.5) * map_data['original_resolution']
        
        return world_x, world_y
    
    def is_cell_free(self, grid_x: int, grid_y: int, map_data: Dict[str, Any], 
                     robot_radius: float = 0.2) -> bool:
        """셀이 로봇이 지나갈 수 있는 자유 공간인지 확인"""
        if (grid_x < 0 or grid_x >= map_data['width'] or 
            grid_y < 0 or grid_y >= map_data['height']):
            return False
        
        # 로봇 반경을 고려한 안전 영역 체크
        radius_cells = int(robot_radius / map_data['resolution']) + 1
        
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                if (check_x < 0 or check_x >= map_data['width'] or 
                    check_y < 0 or check_y >= map_data['height']):
                    continue
                
                # 거리 체크
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    if map_data['map_array'][map_data['height'] - 1 - check_y, check_x] >= self.occupied_threshold * 100:
                        return False
        
        return True
    
    def get_clearance(self, grid_x: int, grid_y: int, map_data: Dict[str, Any]) -> float:
        """해당 위치에서 가장 가까운 장애물까지의 거리 반환"""
        if (grid_x < 0 or grid_x >= map_data['width'] or 
            grid_y < 0 or grid_y >= map_data['height']):
            return 0.0
        
        max_search = 10  # 최대 탐색 반경
        
        for radius in range(1, max_search + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy > radius*radius:
                        continue
                        
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    
                    if (check_x < 0 or check_x >= map_data['width'] or 
                        check_y < 0 or check_y >= map_data['height']):
                        continue
                    
                    if map_data['map_array'][map_data['height'] - 1 - check_y, check_x] >= self.occupied_threshold * 100:
                        return radius * map_data['resolution']
        
        return max_search * map_data['resolution']
