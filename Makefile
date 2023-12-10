CPP = g++

# OpenCV from /build/openvc
#	   -Wl,-rpath=/usr/local/lib64/
CPPFLAGS = -L`pkg-config --variable=libdir opencv4` \
	   `pkg-config --cflags-only-I opencv4` \
	   `pkg-config --libs opencv4` -g3

all: cvfind

cvfind:	cvfind.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

check: cvfind
	./cvfind --test
