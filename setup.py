from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'block_position_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Publishes block position from Gazebo Sim',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'force_publisher = block_position_publisher.force_publisher:main',
            'new_controller = block_position_publisher.geometric_controller:main',
            'gamepad_receiver = block_position_publisher.gamepad1:main',
            'receiver = block_position_publisher.reciever:main',
            
            
        ],
    },
)


