from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow>=2.0",
    "mujoco-py<2.1,>=2.0",
    "metaworld @ git+https://github.com/rlworkgroup/metaworld.git@18118a28c06893da0f363786696cc792457b062b#egg=metaworld",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
]

setup(
    name="continualworld",
    description="Continual World: A Robotic Benchmark For Continual Reinforcement Learning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
