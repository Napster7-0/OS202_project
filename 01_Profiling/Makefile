include ../00_src/Make_osx.inc
#include ../00_src/Make_linux.inc
#include ../00_src/Make_msys2.inc

CXXFLAGS = -std=c++17 -I$(SDL2_PREFIX)/include

ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = $(CXXFLAGS) -O2 -march=native -Wall
CXXFLAGS += -O3 -march=native -Wall
endif

ALL = ant_simu profile_stats

default: help

all: $(ALL)

clean:
	@rm -fr *.o *.exe ant_simu profile_stats *~

.cpp.o:
	$(CXX) $(CXXFLAGS2) -c $< -o $@

ant_simu: ../00_src/ant.o ../00_src/fractal_land.o ../00_src/renderer.o ../00_src/window.o ant_simu.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

profile_stats: profile_stats.o
	$(CXX) $(CXXFLAGS2) $^ -o $@

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@