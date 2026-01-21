cd ~/ros2ws
source install/setup.bash
ros2 launch body_control body.launch.py
sleep 5
cd ~/DEXTRAH/dextrah_lab/tiangong_deployment_scripts
python amazing_hand.py
sleep 2
python tiangong_fabric.py normal
sleep 2