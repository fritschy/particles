OPT = -O3
WARN = -Wall -Wextra
STD = -std=c++11
CXXFLAGS = $(STD) $(OPT) $(WARN) -g -fno-rtti -fno-exceptions
CXX ?= g++
LD = $(CXX)
LIBS = -pthread -lrt -lGL -lGLU -lglut
LDFLAGS = -Wl,-O3 $(LIBS)
CPPFLAGS =

ifneq ($(CROSS),)
CXX := $(CROSS)$(CXX)
STD = -std=c++0x
else
OPT += -march=native
endif

ifeq ($(NOGUI),1)
LIBS = -pthread -lrt
else
CPPFLAGS += -DUSE_GLUT
endif

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

ifeq ($(DEBUG),1)
CXXFLAGS = -O0 -g3 -std=c++11 $(WARN)
CPPFLAGS = -DNO_THREADED_UPDATE
LDFLAGS = $(LIBS) -g3
else
CPPFLAGS += -DNDEBUG
endif

all: particles

particles: particles.o
	$(LD) $(LDFLAGS) -o $@ $^

clean:
	rm -f particles particles.o
