#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>

struct RunningStat {
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();

    void add(double value) {
        sum += value;
        if (value < min) min = value;
        if (value > max) max = value;
    }

    double mean(std::size_t count) const {
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    }
};

int main(int argc, char* argv[]) {
    std::string input_path = "profiling.csv";
    if (argc >= 2) {
        input_path = argv[1];
    }

    std::ifstream in(input_path);
    if (!in) {
        std::cerr << "Erreur: impossible d'ouvrir le fichier '" << input_path << "'.\n";
        return 1;
    }

    std::string header;
    if (!std::getline(in, header)) {
        std::cerr << "Erreur: fichier vide.\n";
        return 1;
    }

    std::size_t count = 0;
    std::size_t first_food_iteration = 0;
    std::size_t food_quantity_final = 0;

    RunningStat ants, evap, update, render, total;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ',');
        std::size_t iteration = static_cast<std::size_t>(std::stoull(token));

        std::getline(ss, token, ',');
        std::size_t food_quantity = static_cast<std::size_t>(std::stoull(token));
        food_quantity_final = food_quantity;

        std::getline(ss, token, ',');
        int first_food_arrived = std::stoi(token);
        if (first_food_arrived == 1 && first_food_iteration == 0) {
            first_food_iteration = iteration;
        }

        std::getline(ss, token, ',');
        double t_ants = std::stod(token);

        std::getline(ss, token, ',');
        double t_evap = std::stod(token);

        std::getline(ss, token, ',');
        double t_update = std::stod(token);

        std::getline(ss, token, ',');
        double t_render = std::stod(token);

        std::getline(ss, token, ',');
        double t_total = std::stod(token);

        ants.add(t_ants);
        evap.add(t_evap);
        update.add(t_update);
        render.add(t_render);
        total.add(t_total);
        ++count;
    }

    if (count == 0) {
        std::cerr << "Erreur: aucune donnée exploitable dans le fichier.\n";
        return 1;
    }

    const double ms = 1e3;
    const double mean_total = total.mean(count);

    auto print_component = [count, ms, mean_total](const std::string& name, const RunningStat& stat) {
        double mean_s = stat.mean(count);
        double share = (mean_total > 0.0) ? (100.0 * mean_s / mean_total) : 0.0;
        std::cout << std::left << std::setw(18) << name
                  << " mean=" << std::setw(10) << std::fixed << std::setprecision(4) << (mean_s * ms) << " ms"
                  << " min=" << std::setw(10) << (stat.min * ms) << " ms"
                  << " max=" << std::setw(10) << (stat.max * ms) << " ms"
                  << " share=" << std::setw(7) << std::setprecision(2) << share << "%\n";
    };

    std::cout << "Profiling lu depuis: " << input_path << "\n";
    std::cout << "Iterations: " << count << "\n";
    if (first_food_iteration > 0) {
        std::cout << "Premiere nourriture au nid: iteration " << first_food_iteration << "\n";
    } else {
        std::cout << "Premiere nourriture au nid: non observee\n";
    }
    std::cout << "Nourriture finale: " << food_quantity_final << "\n\n";

    print_component("ants", ants);
    print_component("evaporation", evap);
    print_component("update", update);
    print_component("render", render);

    std::cout << std::left << std::setw(18) << "total"
              << " mean=" << std::setw(10) << std::fixed << std::setprecision(4) << (total.mean(count) * ms) << " ms"
              << " min=" << std::setw(10) << (total.min * ms) << " ms"
              << " max=" << std::setw(10) << (total.max * ms) << " ms"
              << " share=" << std::setw(7) << "100.00" << "%\n";

    return 0;
}
