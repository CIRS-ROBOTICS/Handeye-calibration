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
        'ur-rtde',
    ],
    entry_points={
        'console_scripts':[

        ]
    },
    # Meta data
    author="Hao, Peng and Wei, Junhang",
    desciption="This is for Robot (UR5)",
    keywords="robot_envs ur5",
    url=""
)
