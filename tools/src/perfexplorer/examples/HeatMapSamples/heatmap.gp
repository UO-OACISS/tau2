set term png size 1024,768
set output "./matmult.ppk-heatmap.png"
set title "Samples in Contexts"
set xlabel "CONTEXT"
set ylabel "SAMPLES"
set tic scale 0
set palette rgbformulae 22,13,10
set xtics (".TAU application" 0,"IMPLICIT_BARRIER: compute [{matmult.c} {86}]" 1,"IMPLICIT_BARRIER: compute_interchange [{matmult.c} {107}]" 2,"PARALLEL_REGION: compute [{matmult.c} {86}]" 3,"PARALLEL_REGION: compute_interchange [{matmult.c} {107}]" 4,"PARALLEL_REGION: compute_triangular [{matmult.c} {131}]" 5,"PARALLEL_REGION: initialize [{matmult_initialize.c} {5}]" 6) rotate by 45 right

set ytics ("UNRESOLVED /usr/lib64/libgomp.so.1.0.0" 0,"compute.omp_fn.2 [{matmult.c} {93}]" 1,"compute.omp_fn.2 [{matmult.c} {97}]" 2,"compute_interchange.omp_fn.1 [{matmult.c} {114}]" 3,"compute_interchange.omp_fn.1 [{matmult.c} {118}]" 4,"compute_triangular.omp_fn.0 [{matmult.c} {138}]" 5,"compute_triangular.omp_fn.0 [{matmult.c} {142}]" 6)

plot '-' with image
0 0 7
0 1 0
0 2 0
0 3 0
0 4 0
0 5 0
0 6 0

1 0 2
1 1 0
1 2 0
1 3 0
1 4 0
1 5 0
1 6 0

2 0 3
2 1 0
2 2 0
2 3 0
2 4 0
2 5 0
2 6 0

3 0 0
3 1 3
3 2 108
3 3 0
3 4 0
3 5 0
3 6 0

4 0 0
4 1 0
4 2 0
4 3 3
4 4 88
4 5 0
4 6 0

5 0 0
5 1 0
5 2 0
5 3 0
5 4 0
5 5 2
5 6 35

6 0 2
6 1 0
6 2 0
6 3 0
6 4 0
6 5 0
6 6 0

