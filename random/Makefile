
CFLAGS  =`pkg-config --cflags --libs opencv`
CC      = g++

all: write.o 

write.o: write.cpp
	$(CC) $(CFLAGS) $< -o $@	

clean:
	rm -rf *.o.*
	rm *.o
	rm *.zip