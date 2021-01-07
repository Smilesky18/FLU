gp:
	icc flu.c gp_ics_version.c detect_sn_ics_version.c dependency_analysis_ics_version.c -O3 -o gp -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp
clean:
	rm -rf gp
