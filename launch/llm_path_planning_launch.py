#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    
    # 패키지 경로
    pkg_share = FindPackageShare('llm_path_planning')
    
    # 파라미터 파일 경로
    params_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'params.yaml'
    ])
    
    # Launch arguments
    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=params_file,
        description='파라미터 파일 경로'
    )
    
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='시뮬레이션 시간 사용 여부'
    )
    
    # LLM Path Planner 노드
    llm_path_planner_node = Node(
        package='llm_path_planning',
        executable='llm_path_planner',
        name='llm_path_planner',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/map', '/map'),
            ('/amcl_pose', '/amcl_pose'), 
            ('/goal_pose', '/goal_pose'),
            ('/candidate_paths', '/candidate_paths'),
            ('/selected_path', '/selected_path'),
        ]
    )
    
    return LaunchDescription([
        declare_params_file_arg,
        declare_use_sim_time_arg,
        llm_path_planner_node,
    ])
