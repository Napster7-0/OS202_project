#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"

// ============================================================
//  Structures pour les mesures de temps
// ============================================================
struct TimingStats {
    double t_ants        = 0.; // Temps passé dans la boucle advance des fourmis
    double t_evaporation = 0.; // Temps passé dans do_evaporation
    double t_update      = 0.; // Temps passé dans phen.update (swap + cl_update)
    double t_render      = 0.; // Temps passé dans le rendu SDL
    std::size_t count    = 0;  // Nombre d'itérations mesurées

    void print() const {
        double total = t_ants + t_evaporation + t_update + t_render;
        std::cout << "\n=== Mesures de temps sur " << count << " itérations ===\n"
                  << "  Fourmis (advance) : " << t_ants        / count * 1e3 << " ms/it  ("
                  << 100.*t_ants/total        << " %)\n"
                  << "  Évaporation       : " << t_evaporation / count * 1e3 << " ms/it  ("
                  << 100.*t_evaporation/total << " %)\n"
                  << "  Update phéromones : " << t_update      / count * 1e3 << " ms/it  ("
                  << 100.*t_update/total      << " %)\n"
                  << "  Rendu SDL         : " << t_render      / count * 1e3 << " ms/it  ("
                  << 100.*t_render/total      << " %)\n"
                  << "  TOTAL             : " << total         / count * 1e3 << " ms/it\n"
                  << std::flush;
    }
    void reset() { t_ants = t_evaporation = t_update = t_render = 0.; count = 0; }
};

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

// SoA : advance_all() remplace la boucle explicite sur vector<ant>
void advance_time( const fractal_land& land, pheronome& phen,
                   const position_t& pos_nest, const position_t& pos_food,
                   ant_colony& ants, std::size_t& cpteur,
                   TimingStats& stats )
{
    auto t0 = Clock::now();
    ants.advance_all( phen, land, pos_food, pos_nest, cpteur );
    auto t1 = Clock::now();
    phen.do_evaporation();
    auto t2 = Clock::now();
    phen.update();
    auto t3 = Clock::now();

    stats.t_ants        += Sec(t1 - t0).count();
    stats.t_evaporation += Sec(t2 - t1).count();
    stats.t_update      += Sec(t3 - t2).count();
    stats.count         += 1;
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
    TimingStats stats;
    const std::size_t PRINT_EVERY = 100; // Affiche les stats toutes les N itérations
    while (cont_loop) {
        ++it;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                cont_loop = false;
        }
        advance_time( land, phen, pos_nest, pos_food, ants, food_quantity, stats );  // SoA

        auto t_render_start = Clock::now();
        renderer.display( win, food_quantity );
        win.blit();
        stats.t_render += Sec(Clock::now() - t_render_start).count();

        if ( not_food_in_nest && food_quantity > 0 ) {
            std::cout << "La première nourriture est arrivée au nid à l'itération " << it << std::endl;
            not_food_in_nest = false;
        }
        if ( it % PRINT_EVERY == 0 ) {
            stats.print();
            stats.reset();
        }
        //SDL_Delay(10);
    }
    SDL_Quit();
    return 0;
}