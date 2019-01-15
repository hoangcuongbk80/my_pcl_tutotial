# my_pcl_tutotial

[Watch video](https://www.youtube.com/watch?v=0gA_Dr9YYRY)

This contains a brief guide how to install / run the code.

## Installation instructions
The tools require full ROS installation. The installation assumes you have Ubuntu 16.04 LTS [ROS Kinetic] or Ubntu 14.04 LTS [ROS Indigo].

## Install ROS:
Please refer to http://wiki.ros.org/kinetic/Installation/Ubuntu

Download the source tree into your catkin workspace (here we assume ~/catkin_ws):

`$ cd ~/catkin_ws/src`

`$ git clone https://github.com/hoangcuongbk80/my_pcl_tutotial.git

## Compile the source

`$ cd ~/catkin_ws`

`$ catkin_make --pkg my_pcl_tutorial

## Then run:

$ roslaunch openni2_launch openni2.launch

![alt text](https://github.com/hoangcuongbk80/my_pcl_tutotial/blob/master/docs/figs/Original.png)

$ rosrun my_pcl_tutorial downsampling

![alt text](https://github.com/hoangcuongbk80/my_pcl_tutotial/blob/master/docs/figs/Dowmsampled.png)

$ rosrun my_pcl_tutorial segmentation

![alt text](https://github.com/hoangcuongbk80/my_pcl_tutotial/blob/master/docs/figs/Segmented.png)
