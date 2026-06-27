Human Pose Estimation and Gesture Classification
Author: Ali Alperen Civil

-------------------------------------------------------------------
DESCRIPTION
-------------------------------------------------------------------
This program recognises body gestures from a live webcam stream using
the k-nearest neighbours (k-NN) method. It is a Python re-implementation
of a MATLAB pose-estimation example used in an Image Processing course.

Six gestures are supported:
  1 - hands down
  2 - hands up
  3 - right hand up
  4 - left hand up
  5 - hands crossed   (new hand gesture)
  6 - leg raised      (new leg gesture)

-------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------
Python 3.x and the following libraries:
  pip install mediapipe opencv-python numpy

Model file (must be placed in the same folder as gesture.py, not
included in this repository because of its size - see .gitignore):
  pose_landmarker_lite.task
Download link:
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

-------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------
1) Teaching mode - record reference poses for your own body:
     python gesture.py teach
   - Stand back so the whole body (head to ankles) is visible.
   - Hold a pose and press a number key (1-6) to save it.
   - Record about 3 samples per gesture.
   - Press 's' to save them to templates.json, ESC to quit.

2) Recognition mode - classify gestures live:
     python gesture.py run
   - The recognised gesture is shown in the top-left corner.
   - Press 's' to save a screenshot, ESC to quit.

Note: templates.json is not included in this repository, since it is
generated from the user's own body measurements during teaching mode.
Run "python gesture.py teach" first to create your own file.

-------------------------------------------------------------------
FILES IN THIS REPOSITORY
-------------------------------------------------------------------
  gesture.py        main program (teaching + recognition)
  pose.py           simple skeleton test script used during setup
  shot_*_*.png      example screenshots of the six recognised gestures
                    (faces blurred for privacy)
  .gitignore        excludes the model file, templates.json and venv/

-------------------------------------------------------------------
PRIVACY NOTE
-------------------------------------------------------------------
The example screenshots have the face blurred before being committed
to this repository. The original (unblurred) screenshots and the full
written report were submitted privately to the course instructor and
are not part of this repository.
