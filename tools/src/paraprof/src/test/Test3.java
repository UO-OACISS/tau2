default.dep
14 templated_functions
%time         msec   total msec       #call      #subrs  usec/call name
0,0,0 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 6000134 30.00 GROUP="TAU_CALLPATH"
 30.0        6,000        6,000           2           0    3000067 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
0,0,0 5 "f3() (sleeps 3 sec)" excl 6000134 30.00 GROUP="TAU_USER"
 30.0        6,000        6,000           2           0    3000067 f3() (sleeps 3 sec)
0,0,0 8 "f5() (sleeps 5 sec)" excl 5000069 25.00 GROUP="TAU_USER"
 25.0        5,000        5,000           1           0    5000069 f5() (sleeps 5 sec)
0,0,0 11 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " excl 5000069 25.00 GROUP="TAU_CALLPATH"
 25.0        5,000        5,000           1           0    5000069 main() (calls f1, f5) => f5() (sleeps 5 sec)  
0,0,0 3 "f2() (sleeps 2 sec, calls f3)" excl 4000295 20.00 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000214 f2() (sleeps 2 sec, calls f3)
0,0,0 2 "f1()   => f4() (sleeps 4 sec, calls f2)  " excl 4000193 20.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000385 f1()   => f4() (sleeps 4 sec, calls f2)  
0,0,0 6 "f4() (sleeps 4 sec, calls f2)" excl 4000193 20.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000385 f4() (sleeps 4 sec, calls f2)
0,0,0 1 "f1()   => f2() (sleeps 2 sec, calls f3)  " excl 2000168 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000237 f1()   => f2() (sleeps 2 sec, calls f3)  
0,0,0 7 "f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " excl 2000127 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000192 f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
0,0,0 0 "f1()  " excl 1000248 5.00 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000870 f1()  
0,0,0 10 "main() (calls f1, f5) => f1()    " excl 1000248 5.00 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000870 main() (calls f1, f5) => f1()    
0,0,0 9 "main() (calls f1, f5)" excl 1944 0.01 GROUP="TAU_DEFAULT"
100.0            1       20,002           1           2   20002883 main() (calls f1, f5)
0,0,0 9 "main() (calls f1, f5)" incl 20002883 100.00 GROUP="TAU_DEFAULT"
100.0            1       20,002           1           2   20002883 main() (calls f1, f5)
0,0,0 10 "main() (calls f1, f5) => f1()    " incl 15000870 74.99 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000870 main() (calls f1, f5) => f1()    
0,0,0 0 "f1()  " incl 15000870 74.99 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000870 f1()  
0,0,0 3 "f2() (sleeps 2 sec, calls f3)" incl 10000429 49.99 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000214 f2() (sleeps 2 sec, calls f3)
0,0,0 2 "f1()   => f4() (sleeps 4 sec, calls f2)  " incl 9000385 45.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000385 f1()   => f4() (sleeps 4 sec, calls f2)  
0,0,0 6 "f4() (sleeps 4 sec, calls f2)" incl 9000385 45.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000385 f4() (sleeps 4 sec, calls f2)
0,0,0 5 "f3() (sleeps 3 sec)" incl 6000134 30.00 GROUP="TAU_USER"
 30.0        6,000        6,000           2           0    3000067 f3() (sleeps 3 sec)
0,0,0 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 6000134 30.00 GROUP="TAU_CALLPATH"
 30.0        6,000        6,000           2           0    3000067 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
0,0,0 1 "f1()   => f2() (sleeps 2 sec, calls f3)  " incl 5000237 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000237 f1()   => f2() (sleeps 2 sec, calls f3)  
0,0,0 7 "f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " incl 5000192 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000192 f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
0,0,0 11 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " incl 5000069 25.00 GROUP="TAU_CALLPATH"
 25.0        5,000        5,000           1           0    5000069 main() (calls f1, f5) => f5() (sleeps 5 sec)  
0,0,0 8 "f5() (sleeps 5 sec)" incl 5000069 25.00 GROUP="TAU_USER"
 25.0        5,000        5,000           1           0    5000069 f5() (sleeps 5 sec)
0,0,1 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 3000070 59.99 GROUP="TAU_CALLPATH"
 60.0        3,000        3,000           1           0    3000070 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
0,0,1 5 "f3() (sleeps 3 sec)" excl 3000070 59.99 GROUP="TAU_USER"
 60.0        3,000        3,000           1           0    3000070 f3() (sleeps 3 sec)
0,0,1 3 "f2() (sleeps 2 sec, calls f3)" excl 2000205 40.00 GROUP="TAU_USER"
100.0        2,000        5,000           1           1    5000275 f2() (sleeps 2 sec, calls f3)
0,0,1 13 "threaded_func() int () => f2() (sleeps 2 sec, calls f3)  " excl 2000205 40.00 GROUP="TAU_CALLPATH"
100.0        2,000        5,000           1           1    5000275 threaded_func() int () => f2() (sleeps 2 sec, calls f3)  
0,0,1 12 "threaded_func() int ()" excl 474 0.01 GROUP="TAU_DEFAULT"
100.0        0.474        5,000           1           1    5000749 threaded_func() int ()
0,0,1 12 "threaded_func() int ()" incl 5000749 100.00 GROUP="TAU_DEFAULT"
100.0        0.474        5,000           1           1    5000749 threaded_func() int ()
0,0,1 3 "f2() (sleeps 2 sec, calls f3)" incl 5000275 99.99 GROUP="TAU_USER"
100.0        2,000        5,000           1           1    5000275 f2() (sleeps 2 sec, calls f3)
0,0,1 13 "threaded_func() int () => f2() (sleeps 2 sec, calls f3)  " incl 5000275 99.99 GROUP="TAU_CALLPATH"
100.0        2,000        5,000           1           1    5000275 threaded_func() int () => f2() (sleeps 2 sec, calls f3)  
0,0,1 5 "f3() (sleeps 3 sec)" incl 3000070 59.99 GROUP="TAU_USER"
 60.0        3,000        3,000           1           0    3000070 f3() (sleeps 3 sec)
0,0,1 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 3000070 59.99 GROUP="TAU_CALLPATH"
 60.0        3,000        3,000           1           0    3000070 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
t 5 "f3() (sleeps 3 sec)" excl 9000204 36.00 GROUP="TAU_USER"
 36.0        9,000        9,000           3           0    3000068 f3() (sleeps 3 sec)
t 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 9000204 36.00 GROUP="TAU_CALLPATH"
 36.0        9,000        9,000           3           0    3000068 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
t 3 "f2() (sleeps 2 sec, calls f3)" excl 6000500 24.00 GROUP="TAU_USER"
 60.0        6,000       15,000           3           3    5000235 f2() (sleeps 2 sec, calls f3)
t 8 "f5() (sleeps 5 sec)" excl 5000069 20.00 GROUP="TAU_USER"
 20.0        5,000        5,000           1           0    5000069 f5() (sleeps 5 sec)
t 11 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " excl 5000069 20.00 GROUP="TAU_CALLPATH"
 20.0        5,000        5,000           1           0    5000069 main() (calls f1, f5) => f5() (sleeps 5 sec)  
t 6 "f4() (sleeps 4 sec, calls f2)" excl 4000193 16.00 GROUP="TAU_USER"
 36.0        4,000        9,000           1           1    9000385 f4() (sleeps 4 sec, calls f2)
t 2 "f1()   => f4() (sleeps 4 sec, calls f2)  " excl 4000193 16.00 GROUP="TAU_CALLPATH"
 36.0        4,000        9,000           1           1    9000385 f1()   => f4() (sleeps 4 sec, calls f2)  
t 13 "threaded_func() int () => f2() (sleeps 2 sec, calls f3)  " excl 2000205 8.00 GROUP="TAU_CALLPATH"
 20.0        2,000        5,000           1           1    5000275 threaded_func() int () => f2() (sleeps 2 sec, calls f3)  
t 1 "f1()   => f2() (sleeps 2 sec, calls f3)  " excl 2000168 8.00 GROUP="TAU_CALLPATH"
 20.0        2,000        5,000           1           1    5000237 f1()   => f2() (sleeps 2 sec, calls f3)  
t 7 "f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " excl 2000127 8.00 GROUP="TAU_CALLPATH"
 20.0        2,000        5,000           1           1    5000192 f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
t 10 "main() (calls f1, f5) => f1()    " excl 1000248 4.00 GROUP="TAU_CALLPATH"
 60.0        1,000       15,000           1           2   15000870 main() (calls f1, f5) => f1()    
t 0 "f1()  " excl 1000248 4.00 GROUP="TAU_USER"
 60.0        1,000       15,000           1           2   15000870 f1()  
t 9 "main() (calls f1, f5)" excl 1944 0.01 GROUP="TAU_DEFAULT"
 80.0            1       20,002           1           2   20002883 main() (calls f1, f5)
t 12 "threaded_func() int ()" excl 474 0.00 GROUP="TAU_DEFAULT"
 20.0        0.474        5,000           1           1    5000749 threaded_func() int ()
t 9 "main() (calls f1, f5)" incl 20002883 80.00 GROUP="TAU_DEFAULT"
 80.0            1       20,002           1           2   20002883 main() (calls f1, f5)
t 0 "f1()  " incl 15000870 59.99 GROUP="TAU_USER"
 60.0        1,000       15,000           1           2   15000870 f1()  
t 10 "main() (calls f1, f5) => f1()    " incl 15000870 59.99 GROUP="TAU_CALLPATH"
 60.0        1,000       15,000           1           2   15000870 main() (calls f1, f5) => f1()    
t 3 "f2() (sleeps 2 sec, calls f3)" incl 15000704 59.99 GROUP="TAU_USER"
 60.0        6,000       15,000           3           3    5000235 f2() (sleeps 2 sec, calls f3)
t 6 "f4() (sleeps 4 sec, calls f2)" incl 9000385 36.00 GROUP="TAU_USER"
 36.0        4,000        9,000           1           1    9000385 f4() (sleeps 4 sec, calls f2)
t 2 "f1()   => f4() (sleeps 4 sec, calls f2)  " incl 9000385 36.00 GROUP="TAU_CALLPATH"
 36.0        4,000        9,000           1           1    9000385 f1()   => f4() (sleeps 4 sec, calls f2)  
t 5 "f3() (sleeps 3 sec)" incl 9000204 36.00 GROUP="TAU_USER"
 36.0        9,000        9,000           3           0    3000068 f3() (sleeps 3 sec)
t 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 9000204 36.00 GROUP="TAU_CALLPATH"
 36.0        9,000        9,000           3           0    3000068 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
t 12 "threaded_func() int ()" incl 5000749 20.00 GROUP="TAU_DEFAULT"
 20.0        0.474        5,000           1           1    5000749 threaded_func() int ()
t 13 "threaded_func() int () => f2() (sleeps 2 sec, calls f3)  " incl 5000275 20.00 GROUP="TAU_CALLPATH"
 20.0        2,000        5,000           1           1    5000275 threaded_func() int () => f2() (sleeps 2 sec, calls f3)  
t 1 "f1()   => f2() (sleeps 2 sec, calls f3)  " incl 5000237 20.00 GROUP="TAU_CALLPATH"
 20.0        2,000        5,000           1           1    5000237 f1()   => f2() (sleeps 2 sec, calls f3)  
t 7 "f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " incl 5000192 20.00 GROUP="TAU_CALLPATH"
 20.0        2,000        5,000           1           1    5000192 f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
t 11 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " incl 5000069 20.00 GROUP="TAU_CALLPATH"
 20.0        5,000        5,000           1           0    5000069 main() (calls f1, f5) => f5() (sleeps 5 sec)  
t 8 "f5() (sleeps 5 sec)" incl 5000069 20.00 GROUP="TAU_USER"
 20.0        5,000        5,000           1           0    5000069 f5() (sleeps 5 sec)
m 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 4500102 36.00 GROUP="TAU_CALLPATH"
 36.0        4,500        4,500         1.5           0    3000068 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
m 5 "f3() (sleeps 3 sec)" excl 4500102 36.00 GROUP="TAU_USER"
 36.0        4,500        4,500         1.5           0    3000068 f3() (sleeps 3 sec)
m 3 "f2() (sleeps 2 sec, calls f3)" excl 3000250 24.00 GROUP="TAU_USER"
 60.0        3,000        7,500         1.5         1.5    5000235 f2() (sleeps 2 sec, calls f3)
m 8 "f5() (sleeps 5 sec)" excl 2500034.5 20.00 GROUP="TAU_USER"
 20.0        2,500        2,500         0.5           0    5000069 f5() (sleeps 5 sec)
m 11 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " excl 2500034.5 20.00 GROUP="TAU_CALLPATH"
 20.0        2,500        2,500         0.5           0    5000069 main() (calls f1, f5) => f5() (sleeps 5 sec)  
m 2 "f1()   => f4() (sleeps 4 sec, calls f2)  " excl 2000096.5 16.00 GROUP="TAU_CALLPATH"
 36.0        2,000        4,500         0.5         0.5    9000385 f1()   => f4() (sleeps 4 sec, calls f2)  
m 6 "f4() (sleeps 4 sec, calls f2)" excl 2000096.5 16.00 GROUP="TAU_USER"
 36.0        2,000        4,500         0.5         0.5    9000385 f4() (sleeps 4 sec, calls f2)
m 13 "threaded_func() int () => f2() (sleeps 2 sec, calls f3)  " excl 1000102.5 8.00 GROUP="TAU_CALLPATH"
 20.0        1,000        2,500         0.5         0.5    5000275 threaded_func() int () => f2() (sleeps 2 sec, calls f3)  
m 1 "f1()   => f2() (sleeps 2 sec, calls f3)  " excl 1000084 8.00 GROUP="TAU_CALLPATH"
 20.0        1,000        2,500         0.5         0.5    5000237 f1()   => f2() (sleeps 2 sec, calls f3)  
m 7 "f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " excl 1000063.5 8.00 GROUP="TAU_CALLPATH"
 20.0        1,000        2,500         0.5         0.5    5000192 f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
m 0 "f1()  " excl 500124 4.00 GROUP="TAU_USER"
 60.0          500        7,500         0.5           1   15000870 f1()  
m 10 "main() (calls f1, f5) => f1()    " excl 500124 4.00 GROUP="TAU_CALLPATH"
 60.0          500        7,500         0.5           1   15000870 main() (calls f1, f5) => f1()    
m 9 "main() (calls f1, f5)" excl 972 0.01 GROUP="TAU_DEFAULT"
 80.0        0.972       10,001         0.5           1   20002883 main() (calls f1, f5)
m 12 "threaded_func() int ()" excl 237 0.00 GROUP="TAU_DEFAULT"
 20.0        0.237        2,500         0.5         0.5    5000749 threaded_func() int ()
m 9 "main() (calls f1, f5)" incl 10001441.5 80.00 GROUP="TAU_DEFAULT"
 80.0        0.972       10,001         0.5           1   20002883 main() (calls f1, f5)
m 10 "main() (calls f1, f5) => f1()    " incl 7500435 59.99 GROUP="TAU_CALLPATH"
 60.0          500        7,500         0.5           1   15000870 main() (calls f1, f5) => f1()    
m 0 "f1()  " incl 7500435 59.99 GROUP="TAU_USER"
 60.0          500        7,500         0.5           1   15000870 f1()  
m 3 "f2() (sleeps 2 sec, calls f3)" incl 7500352 59.99 GROUP="TAU_USER"
 60.0        3,000        7,500         1.5         1.5    5000235 f2() (sleeps 2 sec, calls f3)
m 2 "f1()   => f4() (sleeps 4 sec, calls f2)  " incl 4500192.5 36.00 GROUP="TAU_CALLPATH"
 36.0        2,000        4,500         0.5         0.5    9000385 f1()   => f4() (sleeps 4 sec, calls f2)  
m 6 "f4() (sleeps 4 sec, calls f2)" incl 4500192.5 36.00 GROUP="TAU_USER"
 36.0        2,000        4,500         0.5         0.5    9000385 f4() (sleeps 4 sec, calls f2)
m 4 "f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 4500102 36.00 GROUP="TAU_CALLPATH"
 36.0        4,500        4,500         1.5           0    3000068 f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
m 5 "f3() (sleeps 3 sec)" incl 4500102 36.00 GROUP="TAU_USER"
 36.0        4,500        4,500         1.5           0    3000068 f3() (sleeps 3 sec)
m 12 "threaded_func() int ()" incl 2500374.5 20.00 GROUP="TAU_DEFAULT"
 20.0        0.237        2,500         0.5         0.5    5000749 threaded_func() int ()
m 13 "threaded_func() int () => f2() (sleeps 2 sec, calls f3)  " incl 2500137.5 20.00 GROUP="TAU_CALLPATH"
 20.0        1,000        2,500         0.5         0.5    5000275 threaded_func() int () => f2() (sleeps 2 sec, calls f3)  
m 1 "f1()   => f2() (sleeps 2 sec, calls f3)  " incl 2500118.5 20.00 GROUP="TAU_CALLPATH"
 20.0        1,000        2,500         0.5         0.5    5000237 f1()   => f2() (sleeps 2 sec, calls f3)  
m 7 "f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " incl 2500096 20.00 GROUP="TAU_CALLPATH"
 20.0        1,000        2,500         0.5         0.5    5000192 f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
m 11 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " incl 2500034.5 20.00 GROUP="TAU_CALLPATH"
 20.0        2,500        2,500         0.5           0    5000069 main() (calls f1, f5) => f5() (sleeps 5 sec)  
m 8 "f5() (sleeps 5 sec)" incl 2500034.5 20.00 GROUP="TAU_USER"
 20.0        2,500        2,500         0.5           0    5000069 f5() (sleeps 5 sec)
