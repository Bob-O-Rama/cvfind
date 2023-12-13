#
# Makefile for cvfind - Rev. 20231212_1559
#
CPP = g++

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

CPPFLAGS = `pkg-config --cflags opencv4` -std=c++14 -g3
LDFLAGS = -Wl,-t,--verbose `pkg-config --libs opencv4`

all: cvfind

cvfind.o:	cvfind.cpp
	echo ''
	echo "compile: $(CPP) $^ -o $@ $(CPPFLAGS)"
	echo ''
	$(CPP) $(CPPFLAGS) -c $^ -o $@
	echo ''	

cvfind:	cvfind.o
	echo ''
	echo "link: $(CPP) $(LDFLAGS) $^ -o $@"
	$(CPP) $^ -o $@ $(LDFLAGS)
	echo ''

check: cvfind
	./cvfind --test

clean:
	rm cvfind cvfind.o

install:	cvfind
	echo ''
	echo 'install:'
	install -d $(DESTDIR)$(PREFIX)/bin/
	install -m 755 cvfind $(DESTDIR)$(PREFIX)/bin/
	echo "contents of destdir=$(DESTDIR) prefix=$(PREFIX):" 
	find "$(DESTDIR)$(PREFIX)" -exec ls -lia {} \;
