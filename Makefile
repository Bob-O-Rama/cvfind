CPP = g++
	   
CPPFLAGS = `pkg-config --cflags --libs opencv4` -Wl,-t -g3

all: cvfind

cvfind:	cvfind.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

check: cvfind
	./cvfind --test

clean:
	rm cvfind

