# README.md

# Main file:

		"capture.py"

# Aux/Supporting file:

		"utils.pyx"

# BGR2HSV:

	Implemented both self and using cvtColor.

	Self implementation in utils.pyx:

		function: "bgr2hsv"

# Run:

## Command: 

		"python capture.py -q 'capture' -s 'fast' -o '<path to video file>'"


## usage:
	capture.py [-h] [-q QUERY] [-s SPEED] [-o OUTPUT] [-f FPS]

optional arguments:
  -h, --help            show this help message and exit
  -q QUERY, --query QUERY
                        Path to the query video ('capture' for VideoCapture
                        from webcam)
  -s SPEED, --speed SPEED
                        Which conversion would you prefer ('slow' runs self
                        implemented bgr2hsv and 'fast' runs using cvtColor)
  -o OUTPUT, --output OUTPUT
                        path to output video file (default:
                        'video/capture.avi')
  -f FPS, --fps FPS     FPS of output video (default: 8)
