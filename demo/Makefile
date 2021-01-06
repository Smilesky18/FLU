nicslu:
	icc -O3 demo.c -o nicslu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/
flu:
	icc  main_plan_3.c computing_plan_2_next.c detect_sn.c dependency_analysis.c -O3 -o flu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp -I. -L. -lmetis -lgfortran libhsl_mc64.a	
flu_right:
	icc  flu_main_right_looking.c lu_computation_right_looking.c detect_sn.c dependency_analysis.c -O3 -o flu_right -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp -I. -L. -lmetis -lgfortran libhsl_mc64.a
flu_me:
	icc  flu_main.c computing_plan_2_next.c detect_sn.c dependency_analysis.c double_computing.c computing_plan_2_next_double_column.c -O3 -o flu_me -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp -I. -lgfortran libhsl_mc64.a libmetis.a #-QPARALLEL -L. -lmetis 
flu_double_col:
	icc -g main_plan_3_double_column.c computing_plan_2_next_double_column.c -O3 -o flu_double_col -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp
mkl_pardiso:
	icc -m64  -w -DMKL_ILP64 -I ../include pardiso_unsym_csc_c.c -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -liomp5 -lpthread -lm -ldl -lnicslu -o mkl_pardiso -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/
pardiso:
	icc -O3 pardiso_unsym.c -o pardiso -I ../include -L. -lpardiso600-GNU720-X86-64 -L ~/intel/mkl/lib/intel64 -lmkl_gf_lp64 -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -lmkl_sequential -lmkl_core -lm -lgfortran -fopenmp -I. libhsl_mc64.a libmetis.a
gp:
	icc flu.c gp.c -O3 -o gp -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp
clean:
	rm -rf flu
