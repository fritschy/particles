OPT = -O2 -march=native
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

all: particles

particles: particles.o

clean:
	rm -f particles particles.o
