execute_process(COMMAND "/home/sizhewin/drl_obj_reaching_tracking/build/ur_main/ur_training/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/sizhewin/drl_obj_reaching_tracking/build/ur_main/ur_training/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
