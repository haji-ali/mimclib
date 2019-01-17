#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "var_list.hpp"
#include <chrono>

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
    assert(bases_indices->size() > 0);
    std::uniform_int_distribution<unsigned int> uni_int(0, bases_indices->size()-1);
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
    assert(bases_indices->size() > 0);
    std::uniform_real_distribution<double> uni(0., 1.);

    double acceptanceratio = 1./(4*std::exp(1));
    int count=0;
    for (unsigned int j=0;j<bases_indices->size();j++){
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

extern "C" void max_deg(const VarSizeList* p_basis_list,
                        uint32 dim,
                        ind_t *max_deg){
    std::fill(max_deg, max_deg+dim, 1);
    uint32 basis_count = p_basis_list->size();
    for (uint32 i=0; i<basis_count; i++){
        const mul_ind_t& ind = p_basis_list->get(i);
        for (auto itr = ind.begin(); itr != ind.end(); itr++){
            max_deg[itr->ind] = std::max(max_deg[itr->ind],
                                         static_cast<ind_t>(itr->value + 1));
        }
    }
}

template<typename T, int dim> class ndarray;

template<typename T> class ndarray<T, 2>
{
public:
    static const uint32 dim=2;
    T* data;
    const uint32 I, J;

    ndarray(T* _data, uint32 _I, uint32 _J) : data(_data), I(_I), J(_J){}

    T& operator()(uint32 i, uint32 j){ return data[i*J+j];}
    const T& operator()(uint32 i, uint32 j) const { return data[i*J+j];}
};

template<typename T> class ndarray<T, 3>
{
public:
    static const uint32 dim=3;
    T* data;
    const uint32 I, J, K;

    ndarray(T* _data, uint32 _I, uint32 _J, uint32 _K) : data(_data), I(_I), J(_J), K(_K) {}

    T& operator()(uint32 i, uint32 j, uint32 k) {return data[(i*J + j)*K + k];}
    const T& operator()(uint32 i, uint32 j, uint32 k) const { return data[(i*J + j)*K + k]; }
};

template<typename T>
class array2d {
private:
    array2d(const array2d& rhs);

public:
    typedef T* value_type;
    typedef const T* const_reference;

    array2d() : m_size(0), m_dim(0), m_data(0) {}
    array2d(array2d&& rhs) : m_size(rhs.m_size), m_dim(rhs.m_dim), m_data(rhs.m_data){
        rhs.m_size = rhs.m_dim = 0;
        rhs.m_data = 0;
    }

    array2d& operator=(array2d&& rhs){
        if (this != &rhs){
            delete [] m_data;

            m_size = rhs.m_size;
            m_dim = rhs.m_dim;
            m_data = rhs.m_data;

            rhs.m_size = rhs.m_dim = 0;
            rhs.m_data = 0;
        }
        return *this;
    }

    array2d(size_t _size, size_t _dim) : m_size(_size), m_dim(_dim) {
        m_data = new T[m_size*m_dim];
    }
    ~array2d(){ delete [] m_data;}
    size_t size() const { return m_size; }

    T* operator[](size_t i) { return m_data + i*m_dim;}
    const T* operator[](size_t i) const { return m_data + i*m_dim;}

public:
    size_t m_size;
    size_t m_dim;
    T* m_data;
};

class matvec_data {
public:
    const VarSizeList* p_basis_list;
    ind_t max_deg;
    double* basis_values;
    uint32 dim;
    uint32 pt_count;
    array2d<ind_t> dense_data;
    matvec_data(uint32 _dim) : p_basis_list(NULL), max_deg(1),
                               basis_values(NULL), dim(_dim), pt_count(0){
    }

    ~matvec_data(){
        delete [] basis_values;
    }
};

extern "C" matvec_data* init_matvec(const VarSizeList* p_basis_list,
                                    const double* _X,
                                    uint32 dim,
                                    uint32 pt_count,
                                    bool densify){
    matvec_data *pdata = new matvec_data(dim);
    pdata->pt_count = pt_count;
    pdata->p_basis_list = p_basis_list;

    const VarSizeList& basis_list = *p_basis_list;
    if (densify){
        pdata->dense_data = array2d<ind_t>(basis_list.size(), dim);

        ind_t *data = pdata->dense_data[0];
        uint32 j=0;
        for (uint32 i=0;i<basis_list.size();i++){
            for (uint32 d=0;d<dim;d++){
                ind_t val = basis_list[i][d];
                data[j++] = val;
                pdata->max_deg = std::max(pdata->max_deg,
                                          static_cast<ind_t>(val + 1));
            }
        }

        pdata->basis_values = new double[pdata->dim *   // This in order of speed of change
                                         pdata->max_deg *
                                         pdata->pt_count];

        ndarray<double, 3> basis_values(pdata->basis_values,
                                        pdata->pt_count,
                                        pdata->max_deg,
                                        pdata->dim);
        ndarray<const double, 2> X(_X, pdata->pt_count, pdata->dim);

        for (uint64_t j=0;j<pt_count;j++){
            for (uint64_t d=0;d<dim;d++){
                std::vector<double> vals(pdata->max_deg);
                legendre_pol(X(j, d), pdata->max_deg, -1, 1, &vals[0]);

                for (int i=0;i<pdata->max_deg;i++)
                    basis_values(j, i, d) = vals[i];
            }
        }
    }
    else{
        for (uint32 i=0;i<basis_list.size();i++){
            for (uint32 d=0;d<dim;d++){
                ind_t val = basis_list[i][d];
                pdata->max_deg = std::max(pdata->max_deg,
                                          static_cast<ind_t>(val + 1));
            }
        }
    }
    return pdata;
}

extern "C" void free_matvec(matvec_data *pdata){
    delete pdata;
}

template<bool square, bool transpose, class T>
void matvec_legendre_basis(const T& basis_list,
                           ind_t max_deg,
                           const double *v,  // vector to multiply matrix by. Size: basis_count
                           uint32 dim,
                           uint32 pt_count,
                           double* _basis_values,
                           double *result)   // Output vector, Size: pt_count
{
    uint32 basis_count = basis_list.size();

    std::fill(result, result + (transpose ? basis_count:pt_count), 0);
    ndarray<double, 3> basis_values(_basis_values, pt_count, max_deg, dim);

    for (uint64_t j=0;j<pt_count;j++){
        for (uint64_t i=0; i<basis_count; i++){
            double tmp=1.;
            typename T::const_reference b = basis_list[i];
            for (uint64_t d=0;d<dim;d++)
                tmp *= basis_values(j, b[d], d);

            if (square)
                tmp = std::pow(tmp, 2.0);

            if (transpose) result[i] += v[j]*tmp;
            else           result[j] += v[i]*tmp;
        }
    }
}



template< class T >
void call_matvec_legendre_basis(bool square, bool transpose,
                                const T& basis_list,
                                ind_t max_deg,
                                const double *v,  // vector to multiply matrix by. Size: basis_count
                                uint32 dim,
                                uint32 pt_count,
                                double* basis_values,
                                double *result){
    if (square){
        if (transpose)
            matvec_legendre_basis<true, true>(basis_list, max_deg, v, dim,
                                               pt_count, basis_values, result);
        else
            matvec_legendre_basis<true, false>(basis_list, max_deg, v, dim,
                                               pt_count, basis_values, result);
    }
    else{
        if (transpose)
            matvec_legendre_basis<false, true>(basis_list, max_deg, v, dim,
                                               pt_count, basis_values, result);
        else
            matvec_legendre_basis<false, false>(basis_list, max_deg, v, dim,
                                                pt_count, basis_values, result);
    }
}


extern "C" void matvec_legendre_basis(matvec_data* data,
                                      const double *v,  // vector to multiply matrix by. Size: basis_count
                                      bool square,
                                      bool transpose,
                                      double *result) // Output vector, Size: pt_count
{
    assert(data->dense_data.size() > 0);
    call_matvec_legendre_basis(square, transpose,
                               data->dense_data,
                               data->max_deg, v,
                               data->dim, data->pt_count,
                               data->basis_values, result);
}

extern "C" void assemble_projection_matrix(matvec_data* data, double *_X, double* _result) {
    const VarSizeList& basis_list = *data->p_basis_list;
    const uint32 pt_count = data->pt_count;
    const uint32 basis_count = basis_list.size();
    const ind_t max_deg = data->max_deg;
    const ind_t dim = data->dim;

    ndarray<double, 2> out(_result, pt_count, basis_count);

    ndarray<const double, 2> X(_X, pt_count, dim);

    std::vector<double> basis_values(dim*max_deg);

    for (uint64_t j=0;j<pt_count;j++) {
        for (uint64_t d=0;d<dim;d++){
            legendre_pol(X(j, d), max_deg, -1, 1,
                         &basis_values[d * max_deg]);
        }

        for (uint64_t i=0; i<basis_count; i++){
            typename VarSizeList::const_reference b = basis_list[i];

            double tmp=1.;
            for (uint64_t d=0;d<dim;d++)
                tmp *= basis_values[d*max_deg + b[d]];
            out(j,i) = tmp;
        }
    }
}
/*
#define CLOCK(title, task) {auto t_start = system_clock::now();       \
         task;                                                           \
         auto duration = duration_cast< milliseconds >(system_clock::now() - t_start).count()/1000.; \
         std::cout << title <<  duration << std::endl;}
*/
