// ant.hpp  —  Version 2 : Vectorisation SoA
#ifndef _ANT_HPP_
# define _ANT_HPP_
# include <vector>
# include <cstddef>
# include "pheronome.hpp"
# include "fractal_land.hpp"
# include "basic_types.hpp"

/**
 * @brief Colonie de fourmis en Structure of Arrays (SoA)
 *
 * Au lieu d'un tableau de struct { state, position, seed },
 * on maintient trois tableaux séparés :
 *   m_states    : état de chaque fourmi (0=non chargée, 1=chargée)
 *   m_positions : position (x,y) de chaque fourmi
 *   m_seeds     : graine aléatoire de chaque fourmi
 *
 * Avantage mémoire : quand on itère sur les états ou les positions,
 * les données sont contiguës → meilleure utilisation du cache L1/L2
 * et possibilité de vectorisation SIMD automatique par le compilateur.
 */
class ant_colony
{
public:
    using state_t = int;   // 0 = unloaded, 1 = loaded
    static constexpr state_t unloaded = 0;
    static constexpr state_t loaded   = 1;

    /**
     * @brief Construit une colonie de fourmis
     * @param positions  Positions initiales (une par fourmi)
     * @param seeds      Graines aléatoires (une par fourmi)
     */
    ant_colony( const std::vector<position_t>&  positions,
                const std::vector<std::size_t>& seeds )
        : m_states   ( positions.size(), unloaded ),
          m_positions( positions ),
          m_seeds    ( seeds )
    {}

    std::size_t size() const { return m_positions.size(); }

    static void set_exploration_coef(double eps) { m_eps = eps; }

    /**
     * @brief Fait avancer toutes les fourmis d'un pas de temps
     * @details La boucle externe (indice de fourmi) sera parallélisée
     *          avec OpenMP à l'étape 3 : chaque fourmi est indépendante
     *          des autres pour son propre déplacement.
     */
    void advance_all( pheronome& phen, const fractal_land& land,
                      const position_t& pos_food, const position_t& pos_nest,
                      std::size_t& cpteur_food );

    // Accès direct aux tableaux (utile pour le rendu et les stats)
    const std::vector<position_t>&  positions() const { return m_positions; }
    const std::vector<state_t>&     states()    const { return m_states; }

private:
    static double m_eps;

    // --- Les trois tableaux SoA (layout mémoire plat et contigu) ---
    std::vector<state_t>    m_states;     // États : [s0, s1, s2, ...]
    std::vector<position_t> m_positions;  // Positions : [(x0,y0), (x1,y1), ...]
    std::vector<std::size_t> m_seeds;     // Graines : [seed0, seed1, ...]

    // Avance une seule fourmi (même algorithme que ant::advance v1)
    void advance_one( std::size_t idx,
                      pheronome& phen, const fractal_land& land,
                      const position_t& pos_food, const position_t& pos_nest,
                      std::size_t& cpteur_food );
};

#endif