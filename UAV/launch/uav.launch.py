
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    parameters_file = DeclareLaunchArgument(
        'parameters_file',
        default_value = PathJoinSubstitution(
            [FindPackageShare('UAV'), 'config', 'uav_parameters.yaml']
        ),  
        description='Path to the YAML file with UAV parameters'
    )

    uav_node = Node(
        package='UAV',
        executable='uav_model',
        name='uav',
        parameters=[LaunchConfiguration('parameters_file')],
        remappings=[
            ('control_inputs', '/cf_control/control_command'),
            ('state', '/UAV/state')
        ],
        output='screen'
    )

    return LaunchDescription([
        parameters_file,
        uav_node,
    ])