// ant_simu_mpi.cpp  —  Version 4 : Parallélisation MPI (Approche 1)
//
// STRATÉGIE (Approche 1 du cours) :
//   • Chaque processus MPI possède la carte entière (phéromones + terrain)
//   • Les m fourmis sont réparties équitablement entre les P processus
//     → chaque processus gère m/P fourmis (+ reste sur le dernier)
//   • À chaque pas de temps :
//     1. Chaque processus fait avancer ses fourmis locales
//        (met à jour son buffer_pheronome local via mark_pheronome)
//     2. Évaporation : chaque processus traite les lignes i ∈ [i_start, i_end)
//        de son buffer (parallélisation de la carte entre processus)
//     3. MPI_Allreduce(MPI_MAX) sur tout le buffer pour fusionner les mises
//        à jour de phéromone (cf. énoncé : "valeur la plus grande entre tous
//        les processus") → chaque processus obtient la carte fusionnée
//     4. phen.update() (swap buffer → carte principale)
//
// NOTE : pas de SDL dans cette version (calcul pur), résultats sur stdout
// Compiler : make ant_simu_mpi.exe
// Lancer   : mpirun -np 4 ./ant_simu_mpi.exe

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
#include "rand_generator.hpp"

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nb_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);

    // ── Paramètres de simulation ─────────────────────────────────────────
    const std::size_t nb_ants   = 5000;
    const double      eps       = 0.8;
    const double      alpha     = 0.7;
    const double      beta      = 0.999;
    const int         nb_iter   = 1000; // Nombre de pas de temps à simuler
    const int         LOG_EVERY = 100;  // Fréquence d'affichage des stats
    position_t pos_nest{256, 256};
    position_t pos_food{500, 500};

    // ── Génération du terrain (identique sur tous les processus) ─────────
    fractal_land land(8, 2, 1., 1024);
    double max_val = 0., min_val = 1e18;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j)
            land(i,j) = (land(i,j) - min_val) / delta;

    // ── Répartition des fourmis entre processus ──────────────────────────
    // Processus rank gère les fourmis [first_ant, last_ant)
    std::size_t base    = nb_ants / nb_proc;
    std::size_t rest    = nb_ants % nb_proc;
    std::size_t local_n = base + (static_cast<std::size_t>(rank) < rest ? 1 : 0);
    // Calcul de l'indice de départ de ce processus
    std::size_t first_ant = static_cast<std::size_t>(rank) * base
                          + std::min(static_cast<std::size_t>(rank), rest);

    // Génération des positions et graines initiales reproducibles
    // On fait avancer le générateur global jusqu'à first_ant
    std::size_t seed_global = 2026;
    // Positions pour toutes les fourmis (même graine sur tous les proc → reproducible)
    std::vector<position_t>  all_positions(nb_ants);
    std::vector<std::size_t> all_seeds(nb_ants);
    {
        std::size_t s = seed_global;
        auto gen = [&]() { return rand_int32(0, (int)land.dimensions()-1, s); };
        for (std::size_t i = 0; i < nb_ants; ++i) {
            all_positions[i] = position_t{ gen(), gen() };
            all_seeds[i]     = s;
        }
    }
    // Sous-ensemble local
    std::vector<position_t>  local_pos  ( all_positions.begin() + first_ant,
                                          all_positions.begin() + first_ant + local_n );
    std::vector<std::size_t> local_seeds( all_seeds.begin() + first_ant,
                                          all_seeds.begin() + first_ant + local_n );

    ant_colony::set_exploration_coef(eps);
    ant_colony local_ants( local_pos, local_seeds );

    // ── Phéromones : chaque processus a sa propre copie ──────────────────
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    // Taille du buffer phéromone (stride*stride paires de doubles)
    // On expose le buffer via un pointeur brut pour MPI_Allreduce
    std::size_t dim     = land.dimensions();
    std::size_t stride  = dim + 2;
    std::size_t buf_sz  = stride * stride * 2; // nb de doubles dans le buffer

    // Répartition de l'évaporation : chaque processus traite ses lignes
    std::size_t rows_per_proc = dim / nb_proc;
    std::size_t i_start = 1 + static_cast<std::size_t>(rank) * rows_per_proc;
    std::size_t i_end   = (rank == nb_proc-1) ? dim+1
                          : i_start + rows_per_proc;

    // ── Boucle principale ────────────────────────────────────────────────
    std::size_t local_food  = 0;
    std::size_t global_food = 0;
    double t_ants = 0., t_evap = 0., t_comm = 0.;

    for (int it = 1; it <= nb_iter; ++it) {

        // 1. Avance des fourmis locales (OpenMP si plusieurs threads)
        auto ta0 = Clock::now();
        local_ants.advance_all(phen, land, pos_food, pos_nest, local_food);
        t_ants += Sec(Clock::now() - ta0).count();

        // 2. Évaporation partielle (seulement les lignes de ce processus)
        auto te0 = Clock::now();
        phen.do_evaporation_range(i_start, i_end);
        t_evap += Sec(Clock::now() - te0).count();

        // 3. Synchronisation des buffers phéromone via MPI_Allreduce(MAX)
        //    Chaque processus fusionne son buffer local avec ceux des autres.
        //    L'énoncé précise : "on choisira la valeur la plus grande".
        auto tc0 = Clock::now();
        phen.allreduce_buffer_max();   // wrapper autour de MPI_Allreduce
        t_comm += Sec(Clock::now() - tc0).count();

        // 4. Swap buffer → carte principale
        phen.update();

        // Affichage périodique sur le processus 0
        if (rank == 0 && it % LOG_EVERY == 0) {
            MPI_Reduce(MPI_IN_PLACE, &local_food, 1, MPI_UNSIGNED_LONG,
                       MPI_SUM, 0, MPI_COMM_WORLD);
            global_food = local_food;
            double t_total = t_ants + t_evap + t_comm;
            std::cout << "\n[it=" << it << " | procs=" << nb_proc << "]\n"
                      << "  Fourmis    : " << t_ants / LOG_EVERY * 1e3 << " ms/it\n"
                      << "  Évaporation: " << t_evap / LOG_EVERY * 1e3 << " ms/it\n"
                      << "  Comm MPI   : " << t_comm / LOG_EVERY * 1e3 << " ms/it\n"
                      << "  TOTAL/it   : " << t_total/ LOG_EVERY * 1e3 << " ms/it\n"
                      << "  Nourriture : " << global_food << "\n" << std::flush;
            t_ants = t_evap = t_comm = 0.;
        } else if (rank != 0 && it % LOG_EVERY == 0) {
            MPI_Reduce(&local_food, nullptr, 1, MPI_UNSIGNED_LONG,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0)
        std::cout << "\nSimulation terminée. Nourriture totale : " << global_food << "\n";

    MPI_Finalize();
    return 0;
}
