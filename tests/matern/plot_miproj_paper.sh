#!/bin/bash

# Rule: No arguments
DBHOST="129.67.187.118" # Ultra-magnus
DBHOST="129.67.184.182" # mimic
DB_CONN="-db_engine mysql -db_name mimc -db_host ${DBHOST}"
EST_CMD="OPENBLAS_NUM_THREADS=40 python miproj_esterr.py "
EXAMPLE='sf-kink'
BASETAG="postrev-$EXAMPLE-"

# for nu in 2.5 3.5 4.5 6.5 8.5 10.5
# do
#     echo ./plot_miproj_paper.py $DB_CONN \
#          -db_tag sf-matern-1-$nu -o output/matern-$nu.pdf \
#          -verbose True -all_itr True
# done

for N in 2 3 4 6
do
    #echo $EST_CMD $DB_CONN -db_tag "sf-kink-2-$N%" "&& "
    echo python plot_miproj_paper.py $DB_CONN \
         -db_tag ${BASETAG}2-$N-td-theory -o output/poisson-kink-$N \
         -formats pdf tikz -abs_err \
         -verbose -data_file output/poisson-kink-$N/data.dat # -qoi_exact_tag sf-kink-2-$N-adapt
done
