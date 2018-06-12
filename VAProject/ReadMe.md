# Detection and Tracking

This video analytics project is supposed to track a person with a specific logo shirt in real time.

## Flow of process:

* Every person is tracked using the pedestrian detection in OpenCV

* The tracked person is checked if he/she has the logo on his/her t-shirt.

* If not, a red box is displayed around the person, else a green box highlightes the person.

* In the green box detected, the person's face and eyes are detected and a box is created around them.

* Due to the person having to stand some distance away for the pedestrian detection to work, the eye detection is not proper.
