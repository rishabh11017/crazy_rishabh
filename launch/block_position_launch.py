import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('block_position_publisher')
    world_file = os.path.join(pkg_dir, 'worlds', 'block_world.sdf')
    
    # Start Gazebo Sim
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen',
        name='gz_sim'
    )
    
    # Start the ros_gz_bridge for block pose (delayed to let Gazebo start)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='pose_bridge',
        arguments=['/model/quadcopter/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose'],
        output='screen',
    )
   
    bridge_odom = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='quadcopter_odom_bridge',
        arguments=['/model/quadcopter/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
        output='screen',
    )
    
    # ---------------------------------------------------------
    # NEW: Start the ros_gz_bridge for IMU data
    # ---------------------------------------------------------
    bridge_imu = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='imu_bridge',
        arguments=['/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU'],
        output='screen',
    )
    
    # Start the force publisher node
    force_publisher = Node(
        package='block_position_publisher',
        executable='force_publisher',
        name='force_publisher',
        output='screen',
    )
    
    new_controller = Node(
        package='block_position_publisher',
        executable='new_controller',
        name='new_controller',
        output='screen',
    )   
    gamepad_receiver = Node(
        package='block_position_publisher',
        executable='gamepad_receiver',
        name='gamepad_receiver',
        output='screen',
    )
    bridge_camera = Node(
    package='ros_gz_image',
    executable='image_bridge',
    name='mono_camera_bridge',
    arguments=['/world/movable_shapes_world/model/quadcopter/link/base_link/sensor/mono_camera/image'],
    output='screen',
    )
    bridge1_camera = Node(
    package='ros_gz_image',
    executable='image_bridge',
    name='mono1_camera_bridge',
    arguments=['/world/movable_shapes_world/model/quadcopter/link/base_link/sensor/mono1_camera/image'],
    output='screen',
    )
    
    
    # Added bridge_imu to the actions list here:
    delayed_bridge = TimerAction(
        period=2.0,
        actions=[bridge, force_publisher, bridge_odom, new_controller, bridge_camera,bridge1_camera, gamepad_receiver, bridge_imu]
    )
    
    return LaunchDescription([
        gz_sim,
        delayed_bridge,
    ])
