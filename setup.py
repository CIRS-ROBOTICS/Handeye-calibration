from setuptools import setup, find_packages

setup(
    name="real_lib",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'opencv_python',
        'openni>=2.3.0',
        'open3d',
        'setuptools',
        'pymodbus',
        'math3d',
        'protobuf',
        'ur-rtde==1.4.5',
        'pyrealsense2',
        'open3d'
    ],
    entry_points={
        'console_scripts':[

        ]
    },
    # Meta data
    author="HaoPeng, WeiJunhang and CaoXiaoge ",
    desciption="This is for UR Robot ",
    keywords="robot_envs ur robot",
    url=""
)
