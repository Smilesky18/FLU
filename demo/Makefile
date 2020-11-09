all: nicslu ssn_column_storage_multi_row_computing sn_panel sn_panel_3_v2 sn_panel_3_v3

nicslu_parallel: 
	icc nicslu_parallel.c -O3 -I ../include -L ../lib64 -lnicslu -o nicslu_parallel -Wl,-rpath=../lib64

plan_3:
	icc main_plan_3.c computing_plan_3.c computing_plan_2_next.c computing_plan_2_prior.c  -O3  -I ../include -L ../lib64 -lnicslu -o plan_3 -Wl,-rpath=../lib64 -fopenmp	

plan_3_debug:
	icc -g main_plan_3.c computing_plan_3.c computing_plan_2_next.c computing_plan_2_prior.c  -O3  -I ../include -L ../lib64 -lnicslu -o plan_3_debug -Wl,-rpath=../lib64 -fopenmp	



