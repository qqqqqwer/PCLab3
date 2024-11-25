#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

class HeatDistributionSolver {
private:
    int size;
    double convergence_thresh;
    int max_iterations;
    std::vector<std::vector<double>> current_grid;
    std::vector<std::vector<double>> next_grid;

public:
    HeatDistributionSolver(int n, double thresh = 0.0001, int max_iter = 1000000) 
        : size(n), convergence_thresh(thresh), max_iterations(max_iter) {
        
        current_grid = std::vector<std::vector<double>>(n, std::vector<double>(n, 20.0));
        next_grid = std::vector<std::vector<double>>(n, std::vector<double>(n, 20.0));

        for(int i = 0; i < n; i++) {
            current_grid[0][i] = 100.0;
            next_grid[0][i] = 100.0;
            
            current_grid[n-1][i] = 0.0;
            next_grid[n-1][i] = 0.0;
            
            current_grid[i][0] = 50.0;
            next_grid[i][0] = 50.0;
            
            current_grid[i][n-1] = 50.0;
            next_grid[i][n-1] = 50.0;
        }
    }

    double solve(int num_threads) {
        int iterations = 0;
        double max_diff = 1.0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        omp_set_num_threads(num_threads);
        
        while (max_diff > convergence_thresh && iterations < max_iterations) {
            max_diff = 0.0;
            
            #pragma omp parallel
            {
                double local_max_diff = 0.0;
                
                #pragma omp for collapse(2)
                for(int i = 1; i < size-1; i++) {
                    for(int j = 1; j < size-1; j++) {
                        next_grid[i][j] = (current_grid[i+1][j] + 
                                         current_grid[i-1][j] + 
                                         current_grid[i][j+1] + 
                                         current_grid[i][j-1]) * 0.25;
                                         
                        double diff = std::abs(next_grid[i][j] - current_grid[i][j]);
                        local_max_diff = std::max(local_max_diff, diff);
                    }
                }
                
                #pragma omp critical
                {
                    max_diff = std::max(max_diff, local_max_diff);
                }
            }
            
            current_grid.swap(next_grid);
            iterations++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        std::cout << "Converged after " << iterations << " iterations\n";
        return duration.count();
    }

    void print_grid() {
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                std::cout << current_grid[i][j] << "\t";
            }
            std::cout << "\n";
        }
    }
};

int main() {
    const int grid_sizes[] = {100, 200, 400, 800};
    const int thread_counts[] = {1, 2, 4, 8};
    
    std::cout << "Grid Size\tThreads\tTime(s)\n";
    
    for(int size : grid_sizes) {
        for(int threads : thread_counts) {
            HeatDistributionSolver solver(size);
            double time = solver.solve(threads);
            std::cout << size << "\t\t" << threads << "\t" << time << "\n";
        }
    }
    
    return 0;
}