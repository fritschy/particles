OPT = -O3 -march=native -flto
WARN = -Wall -Wextra
CXXFLAGS = -std=c++11 $(OPT) $(WARN)
CXX ?= g++
LD = $(CXX)
LIBS = -lglut -lGL -lGLU
LDFLAGS = -flto $(LIBS)

all: particles

particles: particles.o

clean:
	rm -f particles particles.o
