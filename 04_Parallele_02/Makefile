include Make_osx.inc
#include ../00_src/Make_linux.inc
#include ../00_src/Make_msys2.inc

CXXFLAGS = -std=c++17 -I$(SDL2_PREFIX)/include $(OMPFLAGS)

ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = $(CXXFLAGS) -O2 -march=native -Wall
endif

ALL = ant_simu ant_simu_mpi

default: help

all: $(ALL)

clean:
	@rm -fr *.o *.exe ant_simu ant_simu_mpi *~

.cpp.o:
	$(CXX) $(CXXFLAGS2) -c $< -o $@

ant_simu: ant.o ../00_src/fractal_land.o renderer.o ../00_src/window.o ant_simu.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

# Version MPI : pas de SDL, uniquement calcul + stats
ant_simu_mpi: ant_mpi.o ../00_src/fractal_land.o ant_simu_mpi.o
	$(MPICXX) $(MPI_CXXFLAGS) $^ -o $@ -lpthread

ant_mpi.o: ant_mpi.cpp ant.hpp pheronome.hpp ../00_src/fractal_land.hpp
	$(MPICXX) $(MPI_CXXFLAGS) -DUSE_MPI -c $< -o $@

ant_simu_mpi.o: ant_simu_mpi.cpp ant.hpp pheronome.hpp ../00_src/fractal_land.hpp
	$(MPICXX) $(MPI_CXXFLAGS) -DUSE_MPI -c $< -o $@

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    MPICXX   :    $(MPICXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@