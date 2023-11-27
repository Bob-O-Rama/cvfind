CPP = g++

# OpenCV from /build/openvc
CPPFLAGS = -L/usr/local/lib64/ \
	   -I/usr/local/include/opencv4 \
	   -Wl,-rpath=/usr/local/lib64/ \
	   `pkg-config --libs opencv4` -g3

all: cvfind

cvfind:	cvfind.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

