from setuptools import find_packages, setup

package_name = "biped_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="phuc",
    maintainer_email="phuc@scout.travel",
    description="rclpy nodes that command the biped through ros2_control.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "squat = biped_control.squat:main",
        ],
    },
)
