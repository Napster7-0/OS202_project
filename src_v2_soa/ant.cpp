// ant.cpp  —  Version 2 : Vectorisation SoA
#include "ant.hpp"
#include "rand_generator.hpp"

double ant_colony::m_eps = 0.;

// =============================================================
//  Avance UNE fourmi (indice idx dans les tableaux SoA)
//  Logique identique à ant::advance() v1, mais les données
//  sont lues/écrites via m_states[idx], m_positions[idx], m_seeds[idx]
// =============================================================
void ant_colony::advance_one( std::size_t idx,
                              pheronome& phen, const fractal_land& land,
                              const position_t& pos_food, const position_t& pos_nest,
                              std::size_t& cpteur_food )
{
    // Lambdas locaux utilisant la graine de la fourmi idx
    auto ant_choice = [&]() { return rand_double( 0., 1.,  m_seeds[idx] ); };
    auto dir_choice = [&]() { return rand_int32 ( 1,  4,  m_seeds[idx] ); };

    double consumed_time = 0.;

    while ( consumed_time < 1. ) {
        int        ind_pher    = ( m_states[idx] == loaded ? 1 : 0 );
        double     choix       = ant_choice();
        position_t old_pos     = m_positions[idx];
        position_t new_pos     = old_pos;

        double max_phen = std::max( { phen( new_pos.x-1, new_pos.y )[ind_pher],
                                      phen( new_pos.x+1, new_pos.y )[ind_pher],
                                      phen( new_pos.x,   new_pos.y-1 )[ind_pher],
                                      phen( new_pos.x,   new_pos.y+1 )[ind_pher] } );

        if ( (choix > m_eps) || (max_phen <= 0.) ) {
            // Exploration aléatoire
            do {
                new_pos = old_pos;
                int d = dir_choice();
                if (d == 1) new_pos.x -= 1;
                if (d == 2) new_pos.y -= 1;
                if (d == 3) new_pos.x += 1;
                if (d == 4) new_pos.y += 1;
            } while ( phen[new_pos][ind_pher] == -1 );
        } else {
            // Suit le gradient de phéromone
            if      ( phen(new_pos.x-1, new_pos.y  )[ind_pher] == max_phen ) new_pos.x -= 1;
            else if ( phen(new_pos.x+1, new_pos.y  )[ind_pher] == max_phen ) new_pos.x += 1;
            else if ( phen(new_pos.x,   new_pos.y-1)[ind_pher] == max_phen ) new_pos.y -= 1;
            else                                                               new_pos.y += 1;
        }

        consumed_time      += land( new_pos.x, new_pos.y );
        phen.mark_pheronome( new_pos );
        m_positions[idx]    = new_pos;

        if ( new_pos == pos_nest ) {
            if ( m_states[idx] == loaded ) cpteur_food += 1;
            m_states[idx] = unloaded;
        }
        if ( new_pos == pos_food ) {
            m_states[idx] = loaded;
        }
    }
}

// =============================================================
//  Boucle sur toutes les fourmis
//  → Prête pour #pragma omp parallel for (étape 3)
// =============================================================
void ant_colony::advance_all( pheronome& phen, const fractal_land& land,
                              const position_t& pos_food, const position_t& pos_nest,
                              std::size_t& cpteur_food )
{
    for ( std::size_t i = 0; i < size(); ++i )
        advance_one( i, phen, land, pos_food, pos_nest, cpteur_food );
}