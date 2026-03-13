include ../00_src/Make_osx.inc
include ../00_src/Make_linux.inc
#include ../00_src/Make_msys2.inc

CXXFLAGS = -std=c++17 -I$(SDL2_PREFIX)/include $(OMPFLAGS)

ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = $(CXXFLAGS) -O2 -march=native -Wall
endif

ALL = ant_simu ant_simu_mpi ant_simu_mpi_domain.exe

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

ant_simu_mpi_domain.exe: ../00_src/fractal_land.o ant_simu_mpi_domain.o
	$(MPICXX) $(MPI_CXXFLAGS) $^ -o $@ -lpthread

ant_mpi.o: ant_mpi.cpp ant.hpp pheronome.hpp ../00_src/fractal_land.hpp
	$(MPICXX) $(MPI_CXXFLAGS) -DUSE_MPI -c $< -o $@

ant_simu_mpi.o: ant_simu_mpi.cpp ant.hpp pheronome.hpp ../00_src/fractal_land.hpp
	$(MPICXX) $(MPI_CXXFLAGS) -DUSE_MPI -c $< -o $@

ant_simu_mpi_domain.o: ant_simu_mpi_domain.cpp ../00_src/fractal_land.hpp ../00_src/rand_generator.hpp
	$(MPICXX) $(MPI_CXXFLAGS) -DUSE_MPI -c $< -o $@

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "    run_domain_profile : run MPI domain profile on np=1,2,4,8"
	@echo "    summary_domain_profile : summarize speedup/efficiency from CSV"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    MPICXX   :    $(MPICXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

run_domain_profile: ant_simu_mpi_domain.exe
	@mkdir -p results_mpi_domain
	@for p in 1 2 4 8 ; do \
		echo "[MPI domain] running np=$$p" ; \
		mpirun -np $$p ./ant_simu_mpi_domain.exe --iters 3000 --ants 5000 \
		  --csv results_mpi_domain/profiling_np$${p}.csv ; \
	done

summary_domain_profile:
	@echo "np,mean_total_ms,speedup,efficiency_pct,mean_ants_ms,mean_evap_ms,mean_comm_border_ms,mean_comm_migrate_ms,mean_update_ms"
	@tmp=$$(mktemp) ; \
	for f in results_mpi_domain/profiling_np*.csv ; do \
		np=$$(echo $$f | sed -E 's/.*profiling_np([0-9]+)\.csv/\1/') ; \
		LC_ALL=C awk -F, 'NR>1{t+=$$7;a+=$$2;e+=$$3;cb+=$$4;cm+=$$5;u+=$$6;n+=1} END{if(n>0) printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", "'"$$np"'", t/n,a/n,e/n,cb/n,cm/n,u/n}' $$f >> $$tmp ; \
	done ; \
	sort -n -t, -k1 $$tmp | LC_ALL=C awk -F, 'BEGIN{t1=0} {np=$$1;tp=$$2;a=$$3;e=$$4;cb=$$5;cm=$$6;u=$$7; if(t1==0) t1=tp; s=t1/tp; eff=100*s/np; printf "%d,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f\n", np,tp,s,eff,a,e,cb,cm,u }' ; \
	rm -f $$tmp

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@