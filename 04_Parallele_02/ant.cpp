// ant.cpp  —  Version 3 : Vectorisation SoA + OpenMP
#include "ant.hpp"
#include "rand_generator.hpp"
#include <omp.h>

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
//  Boucle sur toutes les fourmis — parallélisée avec OpenMP
//
//  Choix de conception :
//  1. La boucle est naturellement parallèle (chaque fourmi est
//     indépendante pour ses données privées : position, état, graine).
//
//  2. cpteur_food : variable partagée mise à jour par plusieurs threads.
//     → On utilise une clause "reduction(+:local_food)" pour éviter
//       les race conditions sans verrou coûteux.
//
//  3. phen.mark_pheronome() : écrit dans m_buffer_pheronome[cellule].
//     → Deux fourmis peuvent écrire sur la même cellule simultanément.
//     → On accepte intentionnellement ce comportement approximatif :
//       l'algorithme ACO est stochastique et converge quand même
//       (cf. énoncé : "on choisira la valeur la plus grande").
//       Pour être strict, on pourrait ajouter #pragma omp critical,
//       mais au prix d'une forte sérialisation.
// =============================================================
void ant_colony::advance_all( pheronome& phen, const fractal_land& land,
                              const position_t& pos_food, const position_t& pos_nest,
                              std::size_t& cpteur_food )
{
    std::size_t local_food = 0;

    #pragma omp parallel for schedule(dynamic, 16) reduction(+:local_food)
    for ( std::size_t i = 0; i < size(); ++i ) {
        advance_one( i, phen, land, pos_food, pos_nest, local_food );
    }

    cpteur_food += local_food;
}