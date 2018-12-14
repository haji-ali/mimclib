#!/bin/bash

EXAMPLE="sf-kink"
BASETAG="postrev-$EXAMPLE-"
DBHOST="129.67.187.118" # Ultra-magnus
DBHOST="129.67.184.182" # mimic
EST_CMD="OPENBLAS_NUM_THREADS=20 python miproj_esterr.py -db_engine mysql -db_name mimc -db_host ${DBHOST}"


# echo $EST_CMD -db_tag "matern_reuse-%10.5%" -qoi_exact_tag matern-reuse-1-10.5
# echo $EST_CMD -db_tag "matern_reuse-%8.5%" -qoi_exact_tag matern-reuse-1-8.5
# echo $EST_CMD -db_tag "matern_reuse-%6.5%" -qoi_exact_tag matern-reuse-1-6.5
# echo $EST_CMD -db_tag "matern_reuse-%4.5%" -qoi_exact_tag matern-reuse-1-4.5
# echo $EST_CMD -db_tag "matern_reuse-%3.5%" -qoi_exact_tag matern-reuse-1-3.5
# echo $EST_CMD -db_tag "matern_reuse-%2.5%" -qoi_exact_tag matern-reuse-1-2.5
# echo $EST_CMD -db_tag "flin-sf-kink-2-1%" -qoi_exact_tag sf-kink-2-1

echo $EST_CMD -db_tag "${BASETAG}2-2% > output/txt/${BASETAG}2-2.txt" # -qoi_exact_tag sf-kink-2-2
echo $EST_CMD -db_tag "${BASETAG}2-3% > output/txt/${BASETAG}2-3.txt" #  -qoi_exact_tag sf-kink-2-3
echo $EST_CMD -db_tag "${BASETAG}2-4% > output/txt/${BASETAG}2-4.txt" #  -qoi_exact_tag sf-kink-2-4
echo $EST_CMD -db_tag "${BASETAG}2-6% > output/txt/${BASETAG}2-6.txt" #  -qoi_exact_tag sf-kink-2-6
