CFLAGS  =`pkg-config --cflags --libs opencv`
CC      = g++

all: proj1a.o proj1b.o

proj1a.o: proj1a.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@

proj1b.o: proj1b.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@