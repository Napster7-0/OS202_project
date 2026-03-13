#pragma once
#include "../00_src/fractal_land.hpp"
#include "ant.hpp"
#include "../00_src/pheronome.hpp"
#include "../00_src/window.hpp"

class Renderer
{
public:
    Renderer(  const fractal_land& land, const pheronome& phen,
               const position_t& pos_nest, const position_t& pos_food,
               const ant_colony& ants );

    Renderer(const Renderer& ) = delete;
    ~Renderer();

    void display( Window& win, std::size_t const& compteur );
private:
    fractal_land const& m_ref_land;
    SDL_Texture* m_land{ nullptr };
    const pheronome& m_ref_phen;
    const position_t& m_pos_nest;
    const position_t& m_pos_food;
    const ant_colony& m_ref_ants;    // ← ant_colony au lieu de vector<ant>
    std::vector<std::size_t> m_curve;
};