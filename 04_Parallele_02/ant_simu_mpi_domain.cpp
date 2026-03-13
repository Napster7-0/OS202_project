#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../00_src/basic_types.hpp"
#include "../00_src/fractal_land.hpp"
#include "../00_src/rand_generator.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using Sec = std::chrono::duration<double>;

constexpr int UNLOADED = 0;
constexpr int LOADED = 1;

struct AntRecord {
    int x;
    int y;
    int state;
    std::uint64_t seed;
};

struct Domain {
    int y_start = 0;
    int y_end = -1;
    int local_h = 0;
    int dim = 0;

    int stride() const { return dim + 2; }
    bool owns_y(int y) const { return y >= y_start && y <= y_end; }
};

struct LocalAnts {
    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> state;
    std::vector<std::uint64_t> seed;

    std::size_t size() const { return x.size(); }

    void reserve(std::size_t n) {
        x.reserve(n);
        y.reserve(n);
        state.reserve(n);
        seed.reserve(n);
    }

    void push(const AntRecord& a) {
        x.push_back(a.x);
        y.push_back(a.y);
        state.push_back(a.state);
        seed.push_back(a.seed);
    }

    AntRecord get(std::size_t i) const {
        return AntRecord{x[i], y[i], state[i], seed[i]};
    }
};

inline Domain compute_domain(int rank, int nproc, int dim) {
    const int base = dim / nproc;
    const int rem = dim % nproc;
    const int rows = base + (rank < rem ? 1 : 0);
    const int y0 = rank * base + std::min(rank, rem);
    const int y1 = y0 + rows - 1;
    return Domain{y0, y1, rows, dim};
}

inline int owner_of_y(int y, int nproc, int dim) {
    const int base = dim / nproc;
    const int rem = dim % nproc;
    const int split = (base + 1) * rem;
    if (base == 0) {
        return std::clamp(y, 0, nproc - 1);
    }
    if (y < split) {
        return y / (base + 1);
    }
    return rem + (y - split) / base;
}

inline std::size_t idx3(int ly, int lx, int ch, int stride) {
    return static_cast<std::size_t>((ly * stride + lx) * 2 + ch);
}

inline int global_y_to_local_row(int gy, const Domain& d) {
    if (gy == d.y_start - 1) return 0;
    if (gy == d.y_end + 1) return d.local_h + 1;
    if (gy < d.y_start || gy > d.y_end) return -1;
    return (gy - d.y_start) + 1;
}

inline double read_pher(const std::vector<double>& map, const Domain& d, int gx, int gy, int ch) {
    if (gx < 0 || gx >= d.dim || gy < 0 || gy >= d.dim) return -1.0;
    const int ly = global_y_to_local_row(gy, d);
    if (ly < 0) return -1.0;
    const int lx = gx + 1;
    return map[idx3(ly, lx, ch, d.stride())];
}

inline void set_boundaries_x(std::vector<double>& field, const Domain& d) {
    const int st = d.stride();
    for (int ly = 0; ly <= d.local_h + 1; ++ly) {
        field[idx3(ly, 0, 0, st)] = -1.0;
        field[idx3(ly, 0, 1, st)] = -1.0;
        field[idx3(ly, d.dim + 1, 0, st)] = -1.0;
        field[idx3(ly, d.dim + 1, 1, st)] = -1.0;
    }
}

inline void set_global_y_borders(std::vector<double>& field, const Domain& d) {
    const int st = d.stride();
    if (d.local_h == 0) return;
    if (d.y_start == 0) {
        for (int lx = 1; lx <= d.dim; ++lx) {
            field[idx3(0, lx, 0, st)] = -1.0;
            field[idx3(0, lx, 1, st)] = -1.0;
        }
    }
    if (d.y_end == d.dim - 1) {
        for (int lx = 1; lx <= d.dim; ++lx) {
            field[idx3(d.local_h + 1, lx, 0, st)] = -1.0;
            field[idx3(d.local_h + 1, lx, 1, st)] = -1.0;
        }
    }
}

inline void seed_sources(std::vector<double>& field, const Domain& d, const position_t& food, const position_t& nest) {
    const int st = d.stride();
    if (d.owns_y(food.y)) {
        const int ly = (food.y - d.y_start) + 1;
        field[idx3(ly, food.x + 1, 0, st)] = 1.0;
    }
    if (d.owns_y(nest.y)) {
        const int ly = (nest.y - d.y_start) + 1;
        field[idx3(ly, nest.x + 1, 1, st)] = 1.0;
    }
}

inline void exchange_ghost_rows(std::vector<double>& map, const Domain& d, int rank, int nproc) {
    const int st = d.stride();
    if (d.local_h == 0) return;

    const int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int down = (rank < nproc - 1) ? rank + 1 : MPI_PROC_NULL;
    const int row_count = st * 2;

    MPI_Sendrecv(
        map.data() + idx3(1, 0, 0, st), row_count, MPI_DOUBLE, up, 100,
        map.data() + idx3(d.local_h + 1, 0, 0, st), row_count, MPI_DOUBLE, down, 100,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        map.data() + idx3(d.local_h, 0, 0, st), row_count, MPI_DOUBLE, down, 101,
        map.data() + idx3(0, 0, 0, st), row_count, MPI_DOUBLE, up, 101,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    set_boundaries_x(map, d);
    set_global_y_borders(map, d);
}

inline void mark_pheromone(std::vector<double>& buffer, const std::vector<double>& map, const Domain& d,
                           int gx, int gy, double alpha, const position_t& food, const position_t& nest) {
    if (!d.owns_y(gy) || gx < 0 || gx >= d.dim) return;

    const int ly = (gy - d.y_start) + 1;
    const int lx = gx + 1;
    const int st = d.stride();

    if (gx == food.x && gy == food.y) {
        buffer[idx3(ly, lx, 0, st)] = 1.0;
    } else {
        const double v1l = std::max(read_pher(map, d, gx - 1, gy, 0), 0.0);
        const double v1r = std::max(read_pher(map, d, gx + 1, gy, 0), 0.0);
        const double v1u = std::max(read_pher(map, d, gx, gy - 1, 0), 0.0);
        const double v1d = std::max(read_pher(map, d, gx, gy + 1, 0), 0.0);
        buffer[idx3(ly, lx, 0, st)] = alpha * std::max({v1l, v1r, v1u, v1d}) +
                                      (1.0 - alpha) * 0.25 * (v1l + v1r + v1u + v1d);
    }

    if (gx == nest.x && gy == nest.y) {
        buffer[idx3(ly, lx, 1, st)] = 1.0;
    } else {
        const double v2l = std::max(read_pher(map, d, gx - 1, gy, 1), 0.0);
        const double v2r = std::max(read_pher(map, d, gx + 1, gy, 1), 0.0);
        const double v2u = std::max(read_pher(map, d, gx, gy - 1, 1), 0.0);
        const double v2d = std::max(read_pher(map, d, gx, gy + 1, 1), 0.0);
        buffer[idx3(ly, lx, 1, st)] = alpha * std::max({v2l, v2r, v2u, v2d}) +
                                      (1.0 - alpha) * 0.25 * (v2l + v2r + v2u + v2d);
    }
}

void advance_local_ants(LocalAnts& ants,
                        std::vector<double>& buffer,
                        const std::vector<double>& map,
                        const Domain& d,
                        const fractal_land& land,
                        const position_t& food,
                        const position_t& nest,
                        double eps,
                        double alpha,
                        int rank,
                        int nproc,
                        std::size_t& local_food,
                        std::vector<std::vector<AntRecord>>& outbound)
{
    LocalAnts survivors;
    survivors.reserve(ants.size());

    for (std::size_t i = 0; i < ants.size(); ++i) {
        int x = ants.x[i];
        int y = ants.y[i];
        int state = ants.state[i];
        std::uint64_t seed = ants.seed[i];

        auto ant_choice = [&]() { return rand_double(0.0, 1.0, seed); };
        auto dir_choice = [&]() { return rand_int32(1, 4, seed); };

        bool migrated = false;
        double consumed = 0.0;

        while (consumed < 1.0) {
            const int ch = (state == LOADED) ? 1 : 0;
            int nx = x;
            int ny = y;

            const double left = read_pher(map, d, x - 1, y, ch);
            const double right = read_pher(map, d, x + 1, y, ch);
            const double up = read_pher(map, d, x, y - 1, ch);
            const double down = read_pher(map, d, x, y + 1, ch);
            const double max_pher = std::max({left, right, up, down});

            const double choice = ant_choice();
            if ((choice > eps) || (max_pher <= 0.0)) {
                do {
                    nx = x;
                    ny = y;
                    const int dmove = dir_choice();
                    if (dmove == 1) nx -= 1;
                    if (dmove == 2) ny -= 1;
                    if (dmove == 3) nx += 1;
                    if (dmove == 4) ny += 1;
                } while (read_pher(map, d, nx, ny, ch) == -1.0);
            } else {
                if (left == max_pher) nx -= 1;
                else if (right == max_pher) nx += 1;
                else if (up == max_pher) ny -= 1;
                else ny += 1;
            }

            consumed += land(static_cast<unsigned long>(nx), static_cast<unsigned long>(ny));
            mark_pheromone(buffer, map, d, nx, ny, alpha, food, nest);

            x = nx;
            y = ny;

            if (x == nest.x && y == nest.y) {
                if (state == LOADED) local_food += 1;
                state = UNLOADED;
            }
            if (x == food.x && y == food.y) {
                state = LOADED;
            }

            const int owner = owner_of_y(y, nproc, d.dim);
            if (owner != rank) {
                outbound[owner].push_back(AntRecord{x, y, state, seed});
                migrated = true;
                break;
            }
        }

        if (!migrated) {
            survivors.x.push_back(x);
            survivors.y.push_back(y);
            survivors.state.push_back(state);
            survivors.seed.push_back(seed);
        }
    }

    ants = std::move(survivors);
}

inline void evaporate(std::vector<double>& buffer, const Domain& d, double beta) {
    const int st = d.stride();
    for (int ly = 1; ly <= d.local_h; ++ly) {
        for (int lx = 1; lx <= d.dim; ++lx) {
            buffer[idx3(ly, lx, 0, st)] *= beta;
            buffer[idx3(ly, lx, 1, st)] *= beta;
        }
    }
}

void migrate_ants_alltoallv(LocalAnts& ants,
                            std::vector<std::vector<AntRecord>>& outbound,
                            MPI_Datatype mpi_ant)
{
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    std::vector<int> send_counts(nproc, 0), recv_counts(nproc, 0);
    for (int r = 0; r < nproc; ++r) send_counts[r] = static_cast<int>(outbound[r].size());

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> send_displs(nproc, 0), recv_displs(nproc, 0);
    std::partial_sum(send_counts.begin(), send_counts.end() - 1, send_displs.begin() + 1);
    std::partial_sum(recv_counts.begin(), recv_counts.end() - 1, recv_displs.begin() + 1);

    const int send_total = std::accumulate(send_counts.begin(), send_counts.end(), 0);
    const int recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    std::vector<AntRecord> send_buf;
    send_buf.reserve(send_total);
    for (int r = 0; r < nproc; ++r) {
        send_buf.insert(send_buf.end(), outbound[r].begin(), outbound[r].end());
    }

    std::vector<AntRecord> recv_buf(recv_total);
    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), mpi_ant,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(), mpi_ant,
                  MPI_COMM_WORLD);

    for (const auto& ant : recv_buf) ants.push(ant);
}

MPI_Datatype create_mpi_ant_type() {
    MPI_Datatype dtype;
    AntRecord dummy{};

    int blocklen[4] = {1, 1, 1, 1};
    MPI_Aint disps[4];
    MPI_Aint base = 0;
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.x, &disps[0]);
    MPI_Get_address(&dummy.y, &disps[1]);
    MPI_Get_address(&dummy.state, &disps[2]);
    MPI_Get_address(&dummy.seed, &disps[3]);
    for (auto& d : disps) d -= base;

    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T};
    MPI_Type_create_struct(4, blocklen, disps, types, &dtype);
    MPI_Type_commit(&dtype);
    return dtype;
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int nb_ants = 5000;
    int nb_iter = 15000;
    const double eps = 0.8;
    const double alpha = 0.7;
    const double beta = 0.999;
    const position_t pos_nest{256, 256};
    const position_t pos_food{500, 500};
    std::string output_csv = "profiling_mpi_domain.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < argc) {
            nb_iter = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--ants" && i + 1 < argc) {
            nb_ants = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--csv" && i + 1 < argc) {
            output_csv = argv[++i];
        }
    }

    fractal_land land(8, 2, 1.0, 1024);
    double max_val = 0.0;
    double min_val = 1e18;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            max_val = std::max(max_val, land(i, j));
            min_val = std::min(min_val, land(i, j));
        }
    }
    const double delta = max_val - min_val;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            land(i, j) = (land(i, j) - min_val) / delta;
        }
    }

    const int dim = static_cast<int>(land.dimensions());
    const Domain dom = compute_domain(rank, nproc, dim);

    MPI_Datatype mpi_ant = create_mpi_ant_type();

    std::vector<AntRecord> scatter_pack;
    std::vector<int> send_counts(nproc, 0), send_displs(nproc, 0);

    if (rank == 0) {
        std::size_t seed = 2026;
        std::vector<std::vector<AntRecord>> buckets(nproc);
        for (int i = 0; i < nb_ants; ++i) {
            const int x = rand_int32(0, dim - 1, seed);
            const int y = rand_int32(0, dim - 1, seed);
            const int owner = owner_of_y(y, nproc, dim);
            buckets[owner].push_back(AntRecord{x, y, UNLOADED, static_cast<std::uint64_t>(seed)});
        }

        for (int r = 0; r < nproc; ++r) send_counts[r] = static_cast<int>(buckets[r].size());
        std::partial_sum(send_counts.begin(), send_counts.end() - 1, send_displs.begin() + 1);

        scatter_pack.reserve(nb_ants);
        for (int r = 0; r < nproc; ++r) {
            scatter_pack.insert(scatter_pack.end(), buckets[r].begin(), buckets[r].end());
        }
    }

    int local_n = 0;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<AntRecord> local_init(local_n);
    MPI_Scatterv(scatter_pack.data(), send_counts.data(), send_displs.data(), mpi_ant,
                 local_init.data(), local_n, mpi_ant, 0, MPI_COMM_WORLD);

    LocalAnts ants;
    ants.reserve(static_cast<std::size_t>(local_n) + 128);
    for (const auto& a : local_init) ants.push(a);

    const std::size_t field_sz = static_cast<std::size_t>(dom.local_h + 2) * static_cast<std::size_t>(dom.dim + 2) * 2;
    std::vector<double> map(field_sz, 0.0);
    std::vector<double> buffer(field_sz, 0.0);

    set_boundaries_x(map, dom);
    set_global_y_borders(map, dom);
    seed_sources(map, dom, pos_food, pos_nest);

    std::size_t local_food = 0;
    std::size_t global_food = 0;
    int first_food_iter = -1;

    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_csv, std::ios::trunc);
        csv << "iteration,ants_ms,evap_ms,comm_border_ms,comm_migrate_ms,update_ms,total_ms,global_food,total_ants\n";
    }

    for (int it = 1; it <= nb_iter; ++it) {
        auto t0 = Clock::now();

        auto tb0 = Clock::now();
        exchange_ghost_rows(map, dom, rank, nproc);
        auto tb1 = Clock::now();

        buffer = map;

        std::vector<std::vector<AntRecord>> outbound(nproc);
        auto ta0 = Clock::now();
        advance_local_ants(ants, buffer, map, dom, land, pos_food, pos_nest,
                           eps, alpha, rank, nproc, local_food, outbound);
        auto ta1 = Clock::now();

        auto te0 = Clock::now();
        evaporate(buffer, dom, beta);
        auto te1 = Clock::now();

        auto tm0 = Clock::now();
        migrate_ants_alltoallv(ants, outbound, mpi_ant);
        auto tm1 = Clock::now();

        auto tu0 = Clock::now();
        map.swap(buffer);
        set_boundaries_x(map, dom);
        set_global_y_borders(map, dom);
        seed_sources(map, dom, pos_food, pos_nest);
        auto tu1 = Clock::now();

        auto t1 = Clock::now();

        std::size_t total_ants_local = ants.size();
        std::size_t total_ants_global = 0;
        MPI_Allreduce(&total_ants_local, &total_ants_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&local_food, &global_food, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (first_food_iter < 0 && global_food > 0) {
            first_food_iter = it;
        }

        const double ants_ms = Sec(ta1 - ta0).count() * 1e3;
        const double evap_ms = Sec(te1 - te0).count() * 1e3;
        const double comm_border_ms = Sec(tb1 - tb0).count() * 1e3;
        const double comm_mig_ms = Sec(tm1 - tm0).count() * 1e3;
        const double update_ms = Sec(tu1 - tu0).count() * 1e3;
        const double total_ms = Sec(t1 - t0).count() * 1e3;

        std::array<double, 6> local_times{ants_ms, evap_ms, comm_border_ms, comm_mig_ms, update_ms, total_ms};
        std::array<double, 6> global_times{};
        MPI_Reduce(local_times.data(), global_times.data(), 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            csv << it << ',' << global_times[0] << ',' << global_times[1] << ','
                << global_times[2] << ',' << global_times[3] << ',' << global_times[4] << ','
                << global_times[5] << ',' << global_food << ',' << total_ants_global << '\n';
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "Profiling MPI domaine écrit dans " << output_csv << "\n";
        std::cout << "Premiere nourriture au nid: iteration " << first_food_iter << "\n";
        std::cout << "Nourriture finale: " << global_food << "\n";
    }

    MPI_Type_free(&mpi_ant);
    MPI_Finalize();
    return 0;
}
