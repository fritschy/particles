OPT = -O3
WARN = -Wall -Wextra
CXXFLAGS = -std=c++0x $(OPT) $(WARN)
CXX ?= g++
LD = $(CXX)
LIBS = -pthread -lrt
LDFLAGS = -Wl,-O3 $(LIBS)
CPPFLAGS = -DNDEBUG

ifneq ($(CROSS),)
CXX := $(CROSS)$(CXX)
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
endif

all: particles

particles: particles.o
	$(LD) $(LDFLAGS) -o $@ $^

clean:
	rm -f particles particles.o
