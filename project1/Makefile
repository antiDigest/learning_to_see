FILES	= proj1a.cpp proj1b.cpp proj1c.cpp proj1d.cpp README.txt images Makefile
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

clean:
	rm -rf *.o.*
	rm *.o
	rm *.zip

zip:
	zip project1.zip $(FILES)

runa:
	./proj1a.o 500 500

runb:
	./proj1b.o 0 0 1 1 input/test0.ppm input/20.png
	./proj1b.o 0.5 0 1 0.5 input/test1.ppm input/21.png
	./proj1b.o 0 0 1 1 input/test2.ppm input/22.png
	./proj1b.o 0 0 0.5 0.5 input/test3.ppm input/23.png
	./proj1b.o 0 0 1 1 input/test4.ppm input/24.png

runc:
	./proj1c.o 0 0 1 1 input/test0.ppm input/30.png
	./proj1c.o 0.5 0 1 0.5 input/test1.ppm input/31.png
	./proj1c.o 0 0 1 1 input/test2.ppm input/32.png
	./proj1c.o 0 0 0.5 0.5 input/test3.ppm input/33.png
	./proj1c.o 0 0 1 1 input/test4.ppm input/34.png

rund:
	./proj1d.o 0 0 1 1 input/test0.ppm input/40.png
	./proj1d.o 0.5 0 1 0.5 input/test1.ppm input/41.png
	./proj1d.o 0 0 1 1 input/test2.ppm input/42.png
	./proj1d.o 0 0 0.5 0.5 input/test3.ppm input/43.png
	./proj1d.o 0 0 1 1 input/test4.ppm input/44.png


# ./proj1b.o 0.3 0.3 0.7 0.7 images/fruits.jpg images/fruits$$(n).png 
# ./proj1b.o 0.3 0.3 0.7 0.7 images/fruits.jpg images/fruits1.bmp
# ./proj1c.o 0.3 0.3 0.7 0.7 images/fruits.jpg images/fruits1.bmp
# ./proj1d.o 0.3 0.3 0.7 0.7 images/fruits.jpg images/fruits1.bmp

# OBJS = MovieList.o Movie.o NameList.o Name.o Iterator.o
# CC = g++
# DEBUG = -g
# CFLAGS = -Wall -c $(DEBUG)
# LFLAGS = -Wall $(DEBUG)

# p1 : $(OBJS)
#     $(CC) $(LFLAGS) $(OBJS) -o p1

# MovieList.o : MovieList.h MovieList.cpp Movie.h NameList.h Name.h Iterator.h
#     $(CC) $(CFLAGS) MovieList.cpp

# Movie.o : Movie.h Movie.cpp NameList.h Name.h
#     $(CC) $(CFLAGS) Movie.cpp

# NameList.o : NameList.h NameList.cpp Name.h 
#     $(CC) $(CFLAGS) NameList.cpp

# Name.o : Name.h Name.cpp 
#     $(CC) $(CFLAGS) Name.cpp

# Iterator.o : Iterator.h Iterator.cpp MovieList.h
#     $(CC) $(CFLAGS) Iterator.cpp

# clean:
#     \rm *.o *~ p1

# tar:
#     tar cfv p1.tar Movie.h Movie.cpp Name.h Name.cpp NameList.h \
#             NameList.cpp  Iterator.cpp Iterator.h