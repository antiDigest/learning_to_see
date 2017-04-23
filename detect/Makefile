CFLAGS  =`pkg-config --cflags --libs opencv`
CC      = clang++

all: detect.o detectWink.o detectSilence.o

detect.o: detect.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@

detectWink.o: detectWink.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@

detectSilence.o: detectSilence.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@

run:
	./detect.o haarcascade_eye.xml

runwink:
	./detectWink.o

runwink2:
	./detectWink.o ./images

silence:
	./detectSilence.o