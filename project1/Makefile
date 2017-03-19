CFLAGS  =`pkg-config --cflags --libs opencv`
CC      = clang++

all: proj1a.o proj1b.o proj1c.o proj1d.o

proj1a.o: proj1a.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@

proj1b.o: proj1b.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@

proj1c.o: proj1c.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@

proj1d.o: proj1d.cpp
	$(CC) -ggdb $(CFLAGS) $< -o $@