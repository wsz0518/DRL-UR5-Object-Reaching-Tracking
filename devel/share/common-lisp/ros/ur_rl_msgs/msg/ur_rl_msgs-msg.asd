
(cl:in-package :asdf)

(defsystem "ur_rl_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "blocks_poses" :depends-on ("_package_blocks_poses"))
    (:file "_package_blocks_poses" :depends-on ("_package"))
  ))