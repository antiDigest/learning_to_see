# README.txt - Project 2

All of these programs have been run and tested on a MAC OS X.

----------------------------------------------------------HOW TO RUN-------------------------------------------------------------

Compiliing the code is simple enough. A makefile is attached with the code. To compile:


		"make"


This will compile both the project files on your system and the following will clean the previously compiled object files


		"make clean"


Running wink on live video:

	
		"make runwink"


Running wink on an image folder named "./images":


		"make runwink2"


Running shush detection on live video:


		"make silence"


Running shush detection on an image folder named "./images":


		"make silence2"


---------------------------------------------------------DOCUMENTATION-----------------------------------------------------------


APPROACH:

My approach takes in account that the eye detector should be strong in order to make a closed eye undetectable when the program runs. The region for eye detection has been limited to 1/5 of face height to 2/5 of face height. Doing this removes any detection which is not in the region of the eyes and hence limits our area of scope.

A similar approach for shush has been done. But, for shush detection, the mouth detection should not be as strong as we do not want half mouth to be detected when a finger is kept on the mouth. Also, the detection region was reduced to 1/5 of face height at the bottom of the face detected.






