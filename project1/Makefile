FILES	= proj1a.cpp proj1b.cpp proj1c.cpp proj1d.cpp README.txt
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