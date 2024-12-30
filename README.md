# Deep Reinforcement Learning based Robot Control Methods for Reaching and Tracking

  Following this README.md, you will set up the workspace and run the Training tasks. In the end you can also save and test your models.
  
  The program is using ROS-Noetic and Python3.

## Requirements and dependencies

#### 1. Install the package under ~/ folder
  ```
  cd ~/
  git clone https://---/---.git
  ```

#### 2. Create same conda environment using yaml:
  ```
  conda env create -f ~/drl_obj_reaching_tracking/src/ur_main/environment.yaml
  ```
#### 3. Install following dependencies
  Here may lack some ROS packages while compiling, please install them according to the error log.
  ```
  sudo apt install ros-noetic-ros-control ros-noetic-ros-controllers
  ```

#### 4. Set ROS workspace path
  Manually find these following yaml files and change the 'user' part of the path:
  
  ```~/drl_obj_reaching_tracking/src/ur_main/ur_reaching/config/ur5_train_params.yaml```

  ```~/drl_obj_reaching_tracking/src/ur_main/ur_tracking/config/training_params.yaml```

#### 4. Compile your workspace
  ```
  cd ~/drl_obj_reaching_tracking/
  catkin_make
  ```

## How to run the Reaching tasks

  Reaching tasks are using Q-Learn and Deep-Q-Network algorithms.

#### launch Q-Learn training process:
  ```
  roslaunch ur_reaching start_training_qlearn_ur5.launch 
  ```

#### launch DQN training program:
  ```
  roslaunch ur_reaching start_training_dqn_ur5.launch 
  ```

## How to run the Tracking tasks

Tracking tasks are using REINFORCE and PPO-GAE algorithms.

#### Firstly, launch the Gazebo and Gym interfaces in one Terminal (can be without conda):
  ```
  roslaunch ur_robotiq_gazebo conveyor_gym.launch
  ```

#### Then, launch the chosen training process in another Terminal:
  - for both:
    ```
    set the z_dist in training_params.yaml
    ```
    
  - for REINFORCE:
    ```
    roslaunch ur_tracking reinforce_main.launch
    ```

  - for PPO-GAE:
    ```
    roslaunch ur_tracking ppo_gae_main.launch
    ```

#### After launching, unpause the Gazebo to start Training:
  - klick the unpause buttom in Gazebo GUI,

  - or type the following command in a new Terminal:
    ```
    rosservice call /gazebo/unpause_physics "{}"
    ```

## How to save and test the trained models

#### Since the save-model-functions are included in main file, if your process finished, it will be automatically saved under:
  - #### for Reaching task
    ```~/drl_obj_reaching_tracking/src/ur_main/ur_reaching/....... ```
  - #### for Tracking task
    ```~/drl_obj_reaching_tracking/src/ur_main/ur_tracking/script/models/ ```
#### And you can test models similar to Training:
  - #### for Reaching task
    In Terminal type the following command for chosen model:
    - for Q-Learn:
      ```
      roslaunch ur_reaching test_qlearn_ur5.launch
      ```
    - for DQN:
      ```
      roslaunch ur_reaching test_dqn_ur5.launch
      ```
  
  - #### for Tracking task

    In one Terminal:
    ```
    roslaunch ur_robotiq_gazebo conveyor_gym.launch
    ```
    In another Terminal:
    - for REINFORCE:
      ```
      roslaunch ur_tracking reinforce_test.launch
      ```
    - for PPO-GAE:
      ```
      roslaunch ur_tracking ppo_gae_test.launch
      ```

### * Important:
- ### if you don't back up your model, it will be overwritten after the next training!
- ### Make sure that you load correct model (check the path of load_model() in ..._test.py)!
