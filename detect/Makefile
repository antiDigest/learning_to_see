FILES	= detectWink.cpp detectSilence.cpp Makefile README.txt
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
	./detectWink.o ./wink

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

zip2:
	zip homework10.zip proj3.py README.txt