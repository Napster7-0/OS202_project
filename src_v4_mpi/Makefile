include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++17
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = CXXFLAGS
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -O3 -march=native -Wall
endif

ALL= ant_simu.exe ant_simu_mpi.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

.cpp.o:
	$(CXX) $(CXXFLAGS2) -c $^ -o $@

ant_simu.exe : ant.o fractal_land.o renderer.o window.o ant_simu.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

# Version MPI : pas de SDL (pas d'affichage), uniquement calcul + stats
ant_simu_mpi.exe : ant_mpi.o fractal_land.o ant_simu_mpi.o
	$(MPICXX) $(CXXFLAGS2) $^ -o $@ -lpthread

ant_mpi.o : ant_mpi.cpp ant.hpp pheronome.hpp fractal_land.hpp
	$(MPICXX) $(CXXFLAGS2) -c $< -o $@

ant_simu_mpi.o : ant_simu_mpi.cpp ant.hpp pheronome.hpp fractal_land.hpp
	$(MPICXX) $(CXXFLAGS2) -c $< -o $@

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
