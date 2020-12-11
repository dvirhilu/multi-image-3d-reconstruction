# Multi-Image 3D Reconstruction - How to Run the Code

**NOTE**: All commands, unless specified otherwise, should be executed in the root directory of this code base

## Setting up the environment
The project uses a virtual environment on a linux machine. The required setup involves two steps:

1. initialize virtual environment
2. install required packages

### initialize virtual environment
1. Ensure that the virtualenv package is installed by running

`
$ pip3 install virtualenv
`

2. Initialize virtual environment by running

`
$ python3 -m venv env/proj_env
`

3. Add required custom modules by creating a file called multi_image_reco.pth in env/proj_env/lib/PythonX.X/site_packages/ and appending the path of the src directory. To find the path of the src directory, navigate to the src directory and run

`
$ pwd
`

4. Step 3 could all be done in the command line if you wish by navigating to the src directory and running

`
$ pwd > ../env/proj_env/lib/PythonX.X/site_packages/multi_image_reco.pth
`

### install required packages
The project uses Python moduels defined in env/requirements.txt, as well as OpenCV and OpenGL

1. OpenCV must be installed from source to be able to use SIFT:
https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/

2. Link the virtual environment to the openCV installation by copying the cv2*.so file to the site-packages folder of the virtual environment. For example:

`
$ cp /usr/lib/pythonX.X/dist-packages/cv2.cpython-36m-aarch64-linux-gnu.so ${REPO_ROOT}/env/proj_env/lib/PythonX.X/site_packages/
`

2. OpenGL can be installed for Python by following these instructions:
https://zoomadmin.com/HowToInstall/UbuntuPackage/python-opengl

3. To install the required python packages, first activate the virtual environment

`
$ source env/proj_env/bin/activate
`

You should see an icon pop up in the command line prompting that the virtual environment has been activated

4. Install required python packages

`
$ pip3 install -r env/requirements.txt
`

5. deactivate the virtual environment

`
$ deactivate
`

## Running the code
Once the environment has been set up, you must run the code within the virtual environment. Activate the environment by running

`
$ source env/proj_env/bin/activate
`

Once the environment has been activated, you should pick the object you would like to reconstruct into a point cloud (some work better than others). There are a variety of options in the images/objects/ directory.

To view the point cloud object using the 3D viewer, run

`
python3 src/main.py -o ${OBJECT_NAME}
`

where object name is the name of the directory in images/objects/ containing the images of the object. Once the 3D viewer pops up, you can rotate the object around using the computer mouse.





