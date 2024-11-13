#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/sizhewin/drl_obj_reaching_tracking/src/obj_tracking/ur_main/ur_reaching"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/sizhewin/drl_obj_reaching_tracking/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/sizhewin/drl_obj_reaching_tracking/install/lib/python3/dist-packages:/home/sizhewin/drl_obj_reaching_tracking/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/sizhewin/drl_obj_reaching_tracking/build" \
    "/usr/bin/python3" \
    "/home/sizhewin/drl_obj_reaching_tracking/src/obj_tracking/ur_main/ur_reaching/setup.py" \
     \
    build --build-base "/home/sizhewin/drl_obj_reaching_tracking/build/obj_tracking/ur_main/ur_reaching" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/sizhewin/drl_obj_reaching_tracking/install" --install-scripts="/home/sizhewin/drl_obj_reaching_tracking/install/bin"
