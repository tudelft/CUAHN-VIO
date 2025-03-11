# CUAHN-VIO: Content-and-uncertainty-aware homography network for visual-inertial odometry

Work published at Robotics and Autonomous Systems [[open-access paper](https://www.sciencedirect.com/science/article/pii/S0921889024002501), [video](https://www.youtube.com/watch?v=_NgDkgON-nE&ab_channel=YingfuXu)]

If you use this code in an academic context, please cite our work:

```bibtex
@article{xu2025cuahn,
  title={CUAHN-VIO: Content-and-uncertainty-aware homography network for visual-inertial odometry},
  author={Xu, Yingfu and de Croon, Guido CHE},
  journal={Robotics and Autonomous Systems},
  volume={185},
  pages={104866},
  year={2025},
  publisher={Elsevier}
}
```

## Usage

### trace_pytorch_model
The Python scripts in this folder were developed using Python==3.9.12, numpy==1.22.4, torch==1.7.1+cu101.

Run `python trace_model.py` to generate the `.pt` model file from the Python `.pth.tar` model file. 

A `.pt` model file is loaded by [libtorch](https://pytorch.org/cppdocs/installing.html) in C++ environment and called in the `HomographyNet` package of `cuahn_ros` to perform neural network inference.

### cuahn_ros
#### Build
cuahn_ros is a ROS 1 project built upon [Commit 83ffb88 of OpenVINS](https://github.com/rpng/open_vins/tree/83ffb88ad35586d86bddc1e041094f4cd3d400df). The required packages are the same as OpenVINS except for [libtorch](https://pytorch.org/cppdocs/installing.html). libtorch==1.7.1+cu101 is installed at `$ENV{HOME}/libtorch` of the developer's laptop computer with cuda 10.1. Please modify the CMAKE_PREFIX_PATH of libtorch according to your libtorch installation at `cuahn_ros/cuahn/CMakeLists.txt` (line 6) and `cuahn_ros/homography_network/CMakeLists.txt` (line 7). 

The developer uses `catkin build` to build the cuahn_ros project. A known issue during building is related to libtorch. The developer uses a workaround as follows. After the first build attempt, error messages containing the following could appear
```
/home/ws/src/cuahn_ros/cuahn/src/state/StateOptions.h:26:10: fatal error: types/LandmarkRepresentation.h: No such file or directory
/home/ws/src/cuahn_ros/cuahn/src/core/VioManagerOptions.h:30:10: fatal error: feat/FeatureInitializerOptions.h: No such file or directory
```
In this case, after the building finishes (`[build] Summary: 5 of 6 packages succeeded.`), comment out `set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch)` at line 6 of `cuahn_ros/cuahn/CMakeLists.txt`, and then `catkin build` again. Package `cuahn` should be built successfully.

#### Run
Modify line 58 of `cuahn_ros/cuahn/launch/uzhfpv.launch` 
```
<param name="network_model_path" type="string" value="$(find HomographyNet)/torch_script_models_laptop/traced_model_3_blocks_using_prior_showError.pt" />
```
to set the path to the `.pt` model file you want to use. If you want to run the full model without using EKF prior, set `false` to line 56 of `uzhfpv.launch`. Otherwise, set it as `true` and set the path to a 3-block model to `network_model_path` (line 58).

Run `roslaunch cuahn uzhfpv.launch` to run CUAHN-VIO on a flight sequence of the UZH-FPV dataset. Set the name and path to the ROS bag of the sequence at lines 8 and 9 of `uzhfpv.launch`.

If the `.pt` model file name contains `showError,` a window displays the photometric error map of the two consecutive images aligned by the network's homography transformation prediction. Set `false` to line 57 of `uzhfpv.launch` to disable video display.
