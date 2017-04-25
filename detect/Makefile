FILES	= detectWink.cpp detectSilence.cpp Makefile
CFLAGS  =`pkg-config --cflags --libs opencv`
CC      = clang++

all: detectWink.o detectSilence.o

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

silence2:
	./detectSilence.o ./shush

clean:
	rm -rf *.o.*
	rm *.o
	rm *.zip

zip:
	zip project2.zip $(FILES)