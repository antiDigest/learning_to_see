
CFLAGS  =`pkg-config --cflags --libs opencv`
CC      = g++

all: write.o 

write.o: write.cpp
	$(CC) $(CFLAGS) $< -o $@	
