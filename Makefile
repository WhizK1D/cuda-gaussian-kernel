CXX=nvcc
CXXFLAGS=-std=c++11 -O0 -x cu

all: a3

clean:
	rm -rf a3
