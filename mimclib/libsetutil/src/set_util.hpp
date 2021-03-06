#ifndef __SET_UTIL_H__
#define __SET_UTIL_H__

typedef int int32;
typedef unsigned int uint32;
typedef unsigned short ind_t;

#ifdef __cplusplus
#include "var_list.hpp"
typedef Node* PTree;
extern "C"{
#else
    typedef void* PProfitCalculator;
    typedef void* PVarSizeList;
#endif

    ind_t GetDefaultSetBase();
    void VarSizeList_check_errors(const PVarSizeList, const double *errors, unsigned char* strange, uint32 count);

    PProfitCalculator CreateMISCProfCalc(ind_t d, ind_t s,
                                         const double *d_rates, const double *s_err_rates);
    PProfitCalculator CreateMIProfCalc(ind_t d, const double *dexp,
                                       double xi, double sexp, double mul);
    PProfitCalculator CreateTDFTProfCalc(ind_t d, const double *td_w, const double *ft_w);
    PProfitCalculator CreateTDHCProfCalc(ind_t d, const double *td_w, const double *hc_w);
    PProfitCalculator CreateMIProjProfCalc(ind_t D, ind_t d,
                                           double beta, double gamma,
                                           double alpha, double theta,
                                           double proj_sample_ratio);

    double GetMinOuterProfit(const PVarSizeList,
                             const PProfitCalculator profCalc,
                             ind_t max_dim);
    void CalculateSetProfit(const PVarSizeList,
                            const PProfitCalculator profCalc,
                            double *log_prof, uint32 size);
    void CheckAdmissibility(const PVarSizeList, ind_t d_start, ind_t d_end,
                            unsigned char *admissible);
    void MakeProfitsAdmissible(const PVarSizeList, ind_t d_start, ind_t d_end,
                               double *pProfits);

    PVarSizeList VarSizeList_expand_set(const PVarSizeList pset,
                                        const double* profit, uint32 count,
                                        uint32 max_added, ind_t dimLookahead);
    PVarSizeList VarSizeList_reduce_set(const PVarSizeList pset,
                                        const ind_t *keep_dim,
                                        ind_t keep_dim_count,
                                        uint32* out_indices,
                                        uint32 out_indices_count);
    PVarSizeList VarSizeList_expand_set_calc(const PVarSizeList pset,
                                             PProfitCalculator profCalc,
                                             double max_prof, ind_t max_d,
                                             double **p_profits);
    PVarSizeList VarSizeList_copy(const PVarSizeList from);
    PVarSizeList VarSizeList_set_diff(const PVarSizeList lhs, const PVarSizeList rhs);
    void VarSizeList_set_union(PVarSizeList lhs, const PVarSizeList rhs,
                               uint32 *ind);

    void VarSizeList_get_adaptive_order(const PVarSizeList pset,
                                        const double *profits,
                                        uint32 *adaptive_order,
                                        uint32 count,
                                        uint32 max_added, ind_t dimLookahead);
    /* void GetLevelBoundaries(const PVarSizeList, const uint32 *levels, */
    /*                         uint32 levels_count, int32 *inner_bnd, */
    /*                         unsigned char *inner_real_lvls); */
    /* void GetBoundaryInd(uint32 setSize, uint32 l, int32 i, */
    /*                     int32* sel, int32* inner_bnd, unsigned char* bnd_ind); */


    void GenTDSet(ind_t d, ind_t base, ind_t *td_set, uint32 count);
    void TensorGrid(ind_t d, ind_t base, const ind_t *m, ind_t* tensor_grid,
                    uint32 count);

    ind_t VarSizeList_max_dim(const PVarSizeList);
    ind_t VarSizeList_get(const PVarSizeList, uint32 i, ind_t* data,
                          ind_t* j, ind_t size);
    ind_t VarSizeList_get_dim(const PVarSizeList, uint32 i);
    ind_t VarSizeList_get_active_dim(const PVarSizeList, uint32 i);
    void VarSizeList_count_neighbors(const PVarSizeList, ind_t *out_neighbors, uint32 count);
    void VarSizeList_is_parent_of_admissible(const PVarSizeList, unsigned char *out_neighbors, uint32 count);
    double VarSizeList_estimate_bias(const PVarSizeList,
                                     const double *err_contributions,
                                     uint32 count,
                                     const double *rates, uint32 rates_size);

    uint32 VarSizeList_count(const PVarSizeList);
    PVarSizeList VarSizeList_sublist(const PVarSizeList,
                                     ind_t d_start, ind_t d_end,
                                     uint32* idx, uint32 _count);
    void VarSizeList_all_dim(const PVarSizeList, uint32 *dim, uint32 size);
    void VarSizeList_all_active_dim(const PVarSizeList, uint32 *active_dim, uint32 size);
    void VarSizeList_to_matrix(const PVarSizeList,
                               ind_t *ij, uint32 ij_size,
                               ind_t *data, uint32 data_size);
    int VarSizeList_find(const PVarSizeList, ind_t *j, ind_t *data,
                         ind_t size);
    PVarSizeList VarSizeList_from_matrix(PVarSizeList,
                                         const ind_t *sizes, uint32 sizes_size,
                                         const ind_t *j, uint32 j_size,
                                         const ind_t *data, uint32 data_size);

    void FreeProfitCalculator(PProfitCalculator profCalc);
    void FreeIndexSet(PVarSizeList);
    void FreeMemory(void **ind_set);

    PTree Tree_new();
    unsigned char Tree_add_node(PTree tree, const double* value, uint32 count, double data, double eps);
    unsigned char Tree_find(PTree tree, const double* value, uint32 count, double* data, unsigned char remove, double eps);
    void Tree_free(PTree tree);

    void Tree_print(PTree tree);
#ifdef __cplusplus
}
#endif

#endif    // __SET_UTIL_H__
