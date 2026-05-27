TARGETS = _kuramoto.so
UNAME := $(shell uname)

all: ${TARGETS}

_kuramoto.so: kuramoto.cpp
ifeq ($(UNAME), Darwin)
	g++ -O3 -fPIC -Wall -std=c++11 -shared -o $@ $^
else
	g++ -O3 -fPIC -Wall -std=c++11 -shared -Wl,-soname,$@ -o $@ $^
endif
