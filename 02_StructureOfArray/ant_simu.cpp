#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <iomanip>
#include "../00_src/fractal_land.hpp"
#include "ant.hpp"
#include "../00_src/pheronome.hpp"
# include "renderer.hpp"
# include "../00_src/window.hpp"
# include "../00_src/rand_generator.hpp"

struct IterationTiming {
    double t_ants        = 0.;
    double t_evaporation = 0.;
    double t_update      = 0.;
    double t_render      = 0.;

    double total() const {
        return t_ants + t_evaporation + t_update + t_render;
    }
};

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

// SoA : advance_all() remplace la boucle explicite sur vector<ant>
void advance_time( const fractal_land& land, pheronome& phen,
                   const position_t& pos_nest, const position_t& pos_food,
                   ant_colony& ants, std::size_t& cpteur,
                   IterationTiming& timing )
{
    auto t0 = Clock::now();
    ants.advance_all( phen, land, pos_food, pos_nest, cpteur );
    auto t1 = Clock::now();
    phen.do_evaporation();
    auto t2 = Clock::now();
    phen.update();
    auto t3 = Clock::now();

    timing.t_ants        = Sec(t1 - t0).count();
    timing.t_evaporation = Sec(t2 - t1).count();
    timing.t_update      = Sec(t3 - t2).count();
}

int main(int nargs, char* argv[])
{
    SDL_Init( SDL_INIT_VIDEO );
    std::size_t seed = 2026; // Graine pour la génération aléatoire ( reproductible )
    const int nb_ants = 5000; // Nombre de fourmis
    const double eps = 0.8;  // Coefficient d'exploration
    const double alpha=0.7; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
    // Location du nid
    position_t pos_nest{256,256};
    // Location de la nourriture
    position_t pos_food{500,500};
    //const int i_food = 500, j_food = 500;    
    // Génération du territoire 512 x 512 ( 2*(2^8) par direction )
    fractal_land land(8,2,1.,1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    /* On redimensionne les valeurs de fractal_land de sorte que les valeurs
    soient comprises entre zéro et un */
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            land(i,j) = (land(i,j)-min_val)/delta;
        }
    // Définition du coefficient d'exploration de toutes les fourmis.
    ant_colony::set_exploration_coef(eps);
    // Génération des positions et graines initiales (SoA)
    auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, (int)land.dimensions()-1, seed); };
    std::vector<position_t>  init_positions;
    std::vector<std::size_t> init_seeds;
    init_positions.reserve(nb_ants);
    init_seeds.reserve(nb_ants);
    for ( int i = 0; i < nb_ants; ++i ) {
        init_positions.push_back( position_t{ gen_ant_pos(), gen_ant_pos() } );
        init_seeds.push_back( seed );
    }
    // Construction de la colonie SoA
    ant_colony ants( init_positions, init_seeds );
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    Window win("Ant Simulation [SoA]", 2*land.dimensions()+10, land.dimensions()+266);
    Renderer renderer( land, phen, pos_nest, pos_food, ants );
    // Compteur de la quantité de nourriture apportée au nid par les fourmis
    size_t food_quantity = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;

    std::string profiling_path = "profiling.csv";
    for ( int i = 1; i < nargs; ++i ) {
        std::string arg = argv[i];
        if ( arg == "--profile" && (i + 1) < nargs ) {
            profiling_path = argv[++i];
        }
    }

    std::ofstream profiling_file(profiling_path, std::ios::out | std::ios::trunc);
    if ( !profiling_file ) {
        std::cerr << "Erreur: impossible d'ouvrir le fichier de profiling '"
                  << profiling_path << "'.\n";
        SDL_Quit();
        return 1;
    }
    profiling_file << std::setprecision(9);
    profiling_file << "iteration,food_quantity,first_food_arrived,t_ants_s,t_evaporation_s,t_update_s,t_render_s,t_total_s\n";

    while (cont_loop) {
        ++it;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                cont_loop = false;
        }
        IterationTiming timing;
        advance_time( land, phen, pos_nest, pos_food, ants, food_quantity, timing );  // SoA

        auto t_render_start = Clock::now();
        renderer.display( win, food_quantity );
        win.blit();
        timing.t_render = Sec(Clock::now() - t_render_start).count();

        bool first_food_arrived = false;
        if ( not_food_in_nest && food_quantity > 0 ) {
            first_food_arrived = true;
            not_food_in_nest = false;
        }

        profiling_file << it << ","
                       << food_quantity << ","
                       << static_cast<int>(first_food_arrived) << ","
                       << timing.t_ants << ","
                       << timing.t_evaporation << ","
                       << timing.t_update << ","
                       << timing.t_render << ","
                       << timing.total() << "\n";

        if ( (it % 100) == 0 ) {
            profiling_file.flush();
        }
        //SDL_Delay(10);
    }
    profiling_file.flush();
    SDL_Quit();
    return 0;
}