OPT = -O3 -march=native
WARN = -Wall -Wextra
CXXFLAGS = -std=c++11 $(OPT) $(WARN)
CXX ?= g++
LD = $(CXX)
LIBS = -lglut -lGL -lGLU
LDFLAGS = -Wl,-O3 $(LIBS)
CPPFLAGS = -DNDEBUG

ifeq ($(PROF),1)
LDFLAGS += -pg
CXXFLAGS += -pg
endif

ifeq ($(LTO),1)
LDFLAGS += -flto
CXXFLAGS += -flto
endif

ifeq ($(OPENMP),1)
LDFLAGS += -fopenmp
CXXFLAGS += -fopenmp
else
ifeq ($(CXX),g++)
LDFLAGS += -fopenmp
CXXFLAGS += -fopenmp
endif
endif

all: particles

particles: particles.o

clean:
	rm -f particles particles.o
