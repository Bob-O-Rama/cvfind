CPP = g++
CPPFLAGS = `pkg-config --cflags opencv4` -g3
LDFLAGS = -Wl,-t `pkg-config --libs opencv4`

all: cvfind

cvfind.o:	cvfind.cpp
	echo ''
	echo "compile: $(CPP) $^ -o $@ $(CPPFLAGS)"
	echo ''
	$(CPP) $^ -c $@ $(CPPFLAGS)
	echo ''	

cvfind:	cvfind.o
	echo ''
	echo "link: $(CPP) $(LDFLAGS) $^ -o $@"
	$(CPP) $(LDFLAGS) $^ -o $@
	echo ''

check: cvfind
	./cvfind --test

clean:
	rm cvfind cvfind.o
