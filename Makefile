CPP = g++

# OpenCV from /build/openvc
# CPPFLAGS = -L`pkg-config --variable=libdir opencv4` \
# 	   `pkg-config --cflags-only-I opencv4` \
#	   -Wl,-rpath=/usr/local/lib64/
# 	   `pkg-config --libs opencv4` -g3
	   
CPPFLAGS = `pkg-config --cflags --libs opencv4`	-g3

all: cvfind

cvfind:	cvfind.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

check: cvfind
	./cvfind --test
