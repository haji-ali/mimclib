#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "var_list.hpp"

static std::mt19937 gen;

void legendre_pol(double X, unsigned int N, double a, double b, double* ret) {
    assert(N>0);
    X = (X-(b+a)/2.) / ((b-a)/2.);
    unsigned int deg = N-1;
    ret[0] = 1.;
    if (N == 1)
        return;
    ret[1] = X;
    for (unsigned int n=1;n<deg;n++){
        ret[n+1] = 1. / (n + 1.) * ((2. * n + 1) * X * ret[n] - n * ret[n - 1]);
    }
    for (unsigned int n=0;n<deg;n++){
        ret[n+1] *= std::sqrt(2*(n+1) + 1);
    }
}

std::vector<double> legendre_pol(double X, unsigned int N, double a, double b) {
    std::vector<double> ret(N);
    legendre_pol(X, N, a, b, &ret[0]);
    return ret;
}


extern "C" unsigned int sample_optimal_random_leg_pts(unsigned int total_N,
                                                      unsigned int* N_per_basis,
                                                      unsigned int max_dim,
                                                      const VarSizeList* bases_indices,
                                                      double *X, double a, double b){
    assert(bases_indices->count() > 0);
    std::uniform_int_distribution<unsigned int> uni_int(0, bases_indices->count()-1);
    std::uniform_real_distribution<double> uni(0., 1.);

    double acceptanceratio = 1./(4*std::exp(1));
    int count=0;
    for (unsigned int j=0;j<total_N;j++){
        auto pol = uni_int(gen);
        N_per_basis[pol]++;
        auto base_pol = bases_indices->get(pol);
        for (unsigned int dim=0;dim<max_dim;dim++){
            bool accept = false;
            double Xreal = 0;
            while (!accept){
                double Xnext = (std::cos(M_PI * uni(gen)) + 1.) / 2.;
                double dens_prop_Xnext = 1. / (M_PI * std::sqrt(Xnext*(1 - Xnext)));
                Xreal = a + Xnext*(b-a);
                double dens_goal_Xnext = legendre_pol(Xreal, 1+base_pol[dim], a, b).back();
                dens_goal_Xnext *= dens_goal_Xnext;
                double alpha = acceptanceratio * dens_goal_Xnext / dens_prop_Xnext;
                double U = uni(gen);
                accept = (U < alpha);
            }
            X[count] = Xreal;
            count++;
        }
    }
    return count;
}

extern "C" unsigned int sample_optimal_leg_pts(const unsigned int *N_per_basis,
                                               unsigned int max_dim,
                                               const VarSizeList* bases_indices,
                                               double *X, double a, double b) {
    assert(bases_indices->count() > 0);
    std::uniform_real_distribution<double> uni(0., 1.);

    double acceptanceratio = 1./(4*std::exp(1));
    int count=0;
    for (unsigned int j=0;j<bases_indices->count();j++){
        auto base_pol = bases_indices->get(j);
        for (unsigned int i=0;i<N_per_basis[j];i++){
            for (unsigned int dim=0;dim<max_dim;dim++){
                bool accept = false;
                double Xreal = 0;
                while (!accept){
                    double Xnext = (std::cos(M_PI * uni(gen)) + 1.) / 2.;
                    double dens_prop_Xnext = 1. / (M_PI * std::sqrt(Xnext*(1 - Xnext)));
                    Xreal = a + Xnext*(b-a);
                    double dens_goal_Xnext = legendre_pol(Xreal, 1+base_pol[dim], a, b).back();
                    dens_goal_Xnext *= dens_goal_Xnext;
                    double alpha = acceptanceratio * dens_goal_Xnext / dens_prop_Xnext;
                    double U = uni(gen);
                    accept = (U < alpha);
                }
                X[count] = Xreal;
                count++;
            }
        }
    }
    return count;
}

extern "C" void evaluate_legendre_basis(const VarSizeList* plist,
                                       uint32 basis_start,
                                       uint32 basis_count,
                                       const double* X,  // pt_count x dim
                                       uint32 dim,
                                       uint32 pt_count,
                                       double* values   // pt_count x basis_count
                                       ) {
    // Find maximum polynomial degree for each dim
    std::vector<ind_t> max_deg(dim, 1);
    for (uint32 i=0; i<basis_count; i++){
        const mul_ind_t& ind = plist->get(basis_start+i);
        for (auto itr = ind.begin(); itr != ind.end(); itr++){
            max_deg[itr->ind] = std::max(max_deg[itr->ind],
                                         static_cast<ind_t>(itr->value + 1));
        }
    }

    // Evaluate leg polynomials up to maximum degree for each point
    std::vector<  std::vector<double> > basis_values(dim);
    for (uint32 d=0;d<dim;d++)
        basis_values[d] = std::vector<double>(pt_count * max_deg[d]);

    // for each dim, d, an array of size count times max_deg[d], evaluating the
    // basis function up max_deg for every point
    for (uint32 j=0;j<pt_count;j++){
        for (uint32 d=0;d<dim;d++){
            legendre_pol(X[j*dim + d],
                         max_deg[d], -1, 1,
                         &(basis_values[d][j*max_deg[d]]));
        }

        for (uint32 i=0; i<basis_count; i++)
            values[ j*basis_count + i ] = 1.;
    }

    for (uint32 j=0;j<pt_count;j++){
        for (uint32 i=0; i<basis_count; i++){
            const mul_ind_t& ind = plist->get(basis_start+i);
            double tmp=1.;
            for (auto itr = ind.begin(); itr != ind.end(); itr++){
                auto d = itr->ind;
                tmp *= basis_values[d][j*max_deg[d] + itr->value];
            }
            values[ j*basis_count + i ] *= tmp;
        }
    }
}
