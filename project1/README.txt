# README.txt

All of these programs have been run and tested on a MAC OS X and so, there is a "Makefile" added in the zip file. To compile all of the files for all questions, simply run:
		
		"make"

A simple "make clean" removes all of the object files and you can run "make" again to compile.

There are specific commands for generating the test images which have been coded into the Makefile and can be run explicitly as stated with each of the questions. The alternate commands are examples

---------------------------------------------------------------------------------------------------------------------------------

proj1a.cpp: This file has the solution to QUESTION 1

To run:

		"make runa"
OR		"./proj1a.o 500 500"

---------------------------------------------------------------------------------------------------------------------------------

proj1b.cpp: This file has the solution to QUESTION 2

To run:

		"make runb"
OR		"./proj1b.o 0 0 1 1 ../images/test0.ppm ../images/20.png"

---------------------------------------------------------------------------------------------------------------------------------

proj1c.cpp: This file has the solution to QUESTION 3

To run:

		"make runc"
OR		"./proj1c.o 0 0 1 1 ../images/test0.ppm ../images/30.png"

---------------------------------------------------------------------------------------------------------------------------------

proj1d.cpp: This file has the solution to QUESTION 4

To run:

		"make rund"
OR		"./proj1d.o 0 0 1 1 ../images/test0.ppm ../images/40.png"

---------------------------------------------------------------------------------------------------------------------------------

Describe the results:

Each of the values in the computation take values of the type "double" and missing any value being of type "double" makes an image look "bad". So, the "double" type during the calculations matters.


