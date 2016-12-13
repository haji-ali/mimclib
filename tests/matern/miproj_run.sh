#!/bin/sh

# make
# rm -f data.sql

COMMON="-mimc_TOL 0.001 -qoi_seed 0 -qoi_problem 0 -qoi_sigma 0.2 -mimc_min_dim 1 -qoi_dim 1 \
       -qoi_x0 0.3 0.4 0.6 -ksp_rtol 1e-25 -ksp_type gmres  -qoi_a0 0 -qoi_f0 1 \
       -qoi_scale 10 -qoi_df_sig 0.5 -mimc_beta 2 -mimc_gamma 1 -mimc_h0inv 3  -mimc_verbose 1 \
       -db True -db_engine sqlite -db_name data.sql"

#python miproj_run.py $COMMON -db_tag "miproj_2.5_nonadapt" -qoi_set_adaptive False &
#python miproj_run.py $COMMON -qoi_df_nu 2.5  -db_tag miproj_2.5_adapt &
python miproj_run.py $COMMON -qoi_df_nu 3.5 -db_tag miproj_adapt &

wait

# ../plot_prog.py -db_engine sqlite -db_name data.sql -db_tag 'miproj_nonadapt' -verbose True -all_itr True &
# ../plot_prog.py -db_engine sqlite -db_name data.sql -db_tag 'miproj_adapt' -verbose True -all_itr True &

wait
