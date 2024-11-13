;; Auto-generated. Do not edit!


(when (boundp 'ur_rl_msgs::blocks_poses)
  (if (not (find-package "UR_RL_MSGS"))
    (make-package "UR_RL_MSGS"))
  (shadow 'blocks_poses (find-package "UR_RL_MSGS")))
(unless (find-package "UR_RL_MSGS::BLOCKS_POSES")
  (make-package "UR_RL_MSGS::BLOCKS_POSES"))

(in-package "ROS")
;;//! \htmlinclude blocks_poses.msg.html


(defclass ur_rl_msgs::blocks_poses
  :super ros::object
  :slots (_x _y _z ))

(defmethod ur_rl_msgs::blocks_poses
  (:init
   (&key
    ((:x __x) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:y __y) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:z __z) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _x __x)
   (setq _y __y)
   (setq _z __z)
   self)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:z
   (&optional __z)
   (if __z (setq _z __z)) _z)
  (:serialization-length
   ()
   (+
    ;; float64[] _x
    (* 8    (length _x)) 4
    ;; float64[] _y
    (* 8    (length _y)) 4
    ;; float64[] _z
    (* 8    (length _z)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _x
     (write-long (length _x) s)
     (dotimes (i (length _x))
       (sys::poke (elt _x i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _y
     (write-long (length _y) s)
     (dotimes (i (length _y))
       (sys::poke (elt _y i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _z
     (write-long (length _z) s)
     (dotimes (i (length _z))
       (sys::poke (elt _z i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _x
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _x (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _x i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _y
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _y (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _y i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _z
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _z (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _z i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get ur_rl_msgs::blocks_poses :md5sum-) "615bdf149c8f78fd88df8f8f89a7e200")
(setf (get ur_rl_msgs::blocks_poses :datatype-) "ur_rl_msgs/blocks_poses")
(setf (get ur_rl_msgs::blocks_poses :definition-)
      "# message type to describe 3-D position of the cylinder blocks
# variable length array, length decided by topic /current_cylinder_blocks
# to be published as a topic

float64[] x  # x coordinate in the world
float64[] y  # y coordinate in the world
float64[] z  # z coordinate in the world
")



(provide :ur_rl_msgs/blocks_poses "615bdf149c8f78fd88df8f8f89a7e200")


