#!/bin/bash

# make
# rm -f data.sql
SEL_CMD="$1"

EXAMPLE='sf-kink'
BASETAG="rev-$EXAMPLE-"
MYSQL_PASSWORD=`gpg2 -q --for-your-eyes-only --no-tty -d ~/.mysql.pass.gpg`
DB_CONN="-db_engine mysql -db_name ah180 -db_host mysql-server-1 -db_user ah180 -db_password \"$MYSQL_PASSWORD\" -mimc_verbose 0 "
# export DB_CONN="-mimc_verbose 10 "

DO_NS='2 3 4 6'

function run_cmd {
    RUN_CMD="OPENBLAS_NUM_THREADS=1 python miproj_run.py -qoi_example $EXAMPLE \
       -mimc_TOL 0 -qoi_seed 0 -mimc_gamma 1 -mimc_h0inv 3 \
       -mimc_bias_calc setutil  $VERBOSE \
       -qoi_seed 0 -ksp_rtol 1e-15 -ksp_type gmres  $DB_CONN "

    echo  $RUN_CMD -miproj_max_lvl $3 \
          -miproj_matfill -qoi_dim $2 -qoi_df_nu $4 \
          ${@:5} -db_tag $BASETAG$2-$4$1
}

function runall_cmd {
    CMN='-qoi_sigma -1 -mimc_beta 1.4142135623730951 -qoi_scale 0.5 '

    for N in $DO_NS
    do
        max_lvl=12
	    ALPHA=3.
	    THETA=`echo "$N/2" | bc -l`

        run_cmd "-theory-discard" 2 $max_lvl $N -miproj_max_vars $N \
                -miproj_s_alpha $ALPHA -miproj_discard_samples \
                -miproj_s_theta $THETA -miproj_d_beta 1. -miproj_d_gamma 1. \
                -miproj_set apriori -mimc_min_dim 1 $CMN -miproj_double_work

        run_cmd "-adapt-discard" 2 $max_lvl $N -miproj_max_vars $N \
                -miproj_set_maxadd 1 -miproj_discard_samples \
                -miproj_set adaptive -mimc_min_dim 1 $CMN

        run_cmd "-adapt-arcsine-discard" 2 $max_lvl $N -miproj_max_vars $N \
                -miproj_set_maxadd 1 -miproj_discard_samples \
                -miproj_pts_sampler arcsine \
                -miproj_set apriori-adapt -mimc_min_dim 1 $CMN

        # run_cmd "-adapt-discard" 2 $max_lvl $N -miproj_max_vars $N \
            #          -miproj_set_maxadd 1 \
            #          -miproj_discard_samples \
            #          -miproj_set apriori-adapt -mimc_min_dim 1 $CMN

        # run_cmd "-adapt-time" 2 $max_lvl $N -miproj_max_vars $N \
            #          -miproj_s_proj_sample_ratio 0. -miproj_set_maxadd 1 \
            #          -miproj_time -miproj_set apriori-adapt -mimc_min_dim 1 $CMN

        # run_cmd "-noproj" 2 $max_lvl $N -miproj_max_vars $N \
            #          -miproj_s_alpha $ALPHA -miproj_s_proj_sample_ratio 0. \
            #          -miproj_set apriori -mimc_min_dim 1 $CMN  -miproj_double_work

        # run_cmd -adapt 2 $max_lvl $N -miproj_max_vars $N -mimc_min_dim 1 \
            #          -miproj_set_maxadd 1 $CMN

        max_lvl=9
        for (( i=0; i<=$max_lvl; i++ ))
        do
            # run_cmd -fix-adapt-$i 2 $(($i+2)) $N -mimc_min_dim 0 -miproj_max_vars $N \
                #          -miproj_fix_lvl $i -miproj_set adaptive \
                #          $CMN

            run_cmd -fix-$i 2 $((($i+2))) $N -mimc_min_dim 0 -miproj_max_vars $N \
                    -miproj_fix_lvl $i -miproj_set apriori $CMN -miproj_double_work \
                    -miproj_discard_samples
        done
    done
}

function plot_cmd {
    for N in $DO_NS
    do
        mkdir -p $OUTPUTDIR/miproj/poisson-kink-$N
        echo python miproj_plot.py $DB_CONN \
             -db_tag ${BASETAG}2-$N-theory-discard -o $OUTPUTDIR/miproj/poisson-kink-$N \
             -formats pdf -abs_err \
             -verbose -data_file $OUTPUTDIR/miproj/data.dat # -qoi_exact_tag sf-kink-2-$N-adapt
    done
}

function est_cmd {
    EST_CMD="OPENBLAS_NUM_THREADS=20 python miproj_esterr.py $DB_CONN"
    mkdir -p $OUTPUTDIR/miproj/log/
    for N in $DO_NS
    do
        echo $EST_CMD -db_tag "${BASETAG}2-${N}% > $OUTPUTDIR/miproj/log/est_${BASETAG}2-${N}.txt" # -qoi_exact_tag sf-kink-2-${N}
    done
}

if [ "$SEL_CMD" = "run" ]; then
    runall_cmd "${@:1}"
elif [ "$SEL_CMD" = "plot" ]; then
    plot_cmd "${@:1}"
elif [ "$SEL_CMD" = "est" ]; then
    est_cmd "${@:1}"
fi;
