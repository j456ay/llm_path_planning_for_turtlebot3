#!/usr/bin/env python3

import json
import requests
from typing import Dict, List, Any
import rclpy
from rclpy.node import Node


class OllamaClient:
    """Ollama REST API 클라이언트"""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "llama3.1:8b"):
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
        
    def generate_path_candidates(self, map_data: Dict[str, Any], start: List[float], 
                               goal: List[float], num_candidates: int = 5) -> Dict[str, Any]:
        """LLM을 사용해 경로 후보들을 생성"""
        
        prompt = self._create_path_planning_prompt(map_data, start, goal, num_candidates)
        
        try:
            response = self._call_ollama(prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"Ollama API 호출 실패: {e}")
            return self._create_fallback_response(start, goal, num_candidates)
    
    def _create_path_planning_prompt(self, map_data: Dict[str, Any], start: List[float], 
                                   goal: List[float], num_candidates: int) -> str:
        """경로 계획을 위한 프롬프트 생성"""
        
        prompt = f"""
You are an expert path planning AI for a mobile robot. Given the following occupancy grid map and start/goal positions, generate {num_candidates} different path candidates as sequences of waypoints.

Map Information:
- Width: {map_data['width']} cells
- Height: {map_data['height']} cells  
- Resolution: {map_data['resolution']} meters/cell
- Origin: [{map_data['origin_x']}, {map_data['origin_y']}]
- Occupied cells (obstacles): {map_data['blocked_cells'][:50]}...  # 처음 50개만 표시

Start Position: [{start[0]:.2f}, {start[1]:.2f}] (grid coordinates)
Goal Position: [{goal[0]:.2f}, {goal[1]:.2f}] (grid coordinates)

Requirements:
1. Generate {num_candidates} different path candidates
2. Each path should be a sequence of waypoints in grid coordinates [x, y]
3. Avoid occupied cells and maintain safe clearance from obstacles
4. Consider different strategies: direct path, wall-following, wide detours, etc.
5. Waypoints should be spaced reasonably (not too dense, not too sparse)

Output format (JSON only, no additional text):
{{
  "candidates": [
    {{
      "id": "C1",
      "waypoints": [[x1, y1], [x2, y2], ..., [{goal[0]:.1f}, {goal[1]:.1f}]],
      "strategy": "direct_path",
      "rationale": "Short description of the path strategy"
    }},
    {{
      "id": "C2", 
      "waypoints": [[x1, y1], [x2, y2], ..., [{goal[0]:.1f}, {goal[1]:.1f}]],
      "strategy": "safe_detour",
      "rationale": "Short description of the path strategy"
    }}
  ]
}}

Generate diverse paths with different risk/reward tradeoffs.
"""
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Ollama API 호출"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,  # 적당한 창의성
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답을 파싱"""
        try:
            # JSON 부분만 추출 시도
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("JSON 형식을 찾을 수 없음")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"LLM 응답 파싱 실패: {e}")
            print(f"Response: {response[:500]}...")
            return {"candidates": []}
    
    def _create_fallback_response(self, start: List[float], goal: List[float], 
                                num_candidates: int) -> Dict[str, Any]:
        """LLM 호출 실패시 폴백 응답"""
        candidates = []
        
        for i in range(num_candidates):
            # 간단한 직선 경로 + 약간의 변형
            waypoints = [start]
            
            # 중간 지점들 추가
            steps = 3 + i  # 각 후보마다 다른 step 수
            for j in range(1, steps):
                t = j / steps
                x = start[0] + t * (goal[0] - start[0]) + (i - 2) * 2  # 약간씩 다른 경로
                y = start[1] + t * (goal[1] - start[1]) + (i % 2) * 2
                waypoints.append([round(x, 2), round(y, 2)])
            
            waypoints.append(goal)
            
            candidates.append({
                "id": f"C{i+1}",
                "waypoints": waypoints,
                "strategy": "fallback_path",
                "rationale": f"Fallback candidate {i+1}"
            })
        
        return {"candidates": candidates}
