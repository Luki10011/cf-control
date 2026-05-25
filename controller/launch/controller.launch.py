import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    controller_params = DeclareLaunchArgument(
        'controller_params',
        default_value=PathJoinSubstitution(
            [FindPackageShare('controller'), 'config', 'uav_controller_params.yaml']
        ),
        description='Path to the controller YAML parameter file.',
    )

    gazebo_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_crazyflie_bringup'),
                'launch',
                'crazyflie_simulation.launch.py',
            )
        ),
    )

    uav_model = Node(
        package='UAV',
        executable='uav_model',
        name='uav',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('UAV'),
                'config',
                'uav_parameters.yaml',
            ])
        ],
        remappings=[
            ('control_inputs', '/cf_control/control_command'),
            ('state', '/UAV/state'),
        ],
        output='screen',
    )

    controller_node = Node(
        package='controller',
        executable='controller_node',
        name='controller_node',
        parameters=[LaunchConfiguration('controller_params')],
        remappings=[
            ('state', '/UAV/state'),
            ('control_inputs', '/cf_control/control_command'),
        ],
        output='screen',
    )

    return LaunchDescription([
        controller_params,
        gazebo_bringup,
        uav_model,
        controller_node,
    ])
