default.dep
13 templated_functions
%time         msec   total msec       #call      #subrs  usec/call name
0,0,0 2 "f3() (sleeps 3 sec)" excl 5999972 30.00 GROUP="TAU_USER"
 30.0        5,999        5,999           2           0    2999986 f3() (sleeps 3 sec)
0,0,0 4 "f5() (sleeps 5 sec)" excl 4999923 25.00 GROUP="TAU_USER"
 25.0        4,999        4,999           1           0    4999923 f5() (sleeps 5 sec)
0,0,0 12 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " excl 4999923 25.00 GROUP="TAU_CALLPATH"
 25.0        4,999        4,999           1           0    4999923 main() (calls f1, f5) => f5() (sleeps 5 sec)  
0,0,0 1 "f2() (sleeps 2 sec, calls f3)" excl 4000342 20.00 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000157 f2() (sleeps 2 sec, calls f3)
0,0,0 9 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  " excl 4000046 20.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000213 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  
0,0,0 3 "f4() (sleeps 4 sec, calls f2)" excl 4000046 20.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000213 f4() (sleeps 4 sec, calls f2)
0,0,0 11 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 2999988 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999988 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
0,0,0 8 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 2999984 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999984 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
0,0,0 10 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " excl 2000179 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000167 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
0,0,0 7 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  " excl 2000163 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000147 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  
0,0,0 6 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  " excl 1000278 5.00 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000638 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  
0,0,0 0 "f1() (sleeps 1 sec, calls f2, f4)" excl 1000278 5.00 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000638 f1() (sleeps 1 sec, calls f2, f4)
0,0,0 5 "main() (calls f1, f5)" excl 1023 0.01 GROUP="TAU_DEFAULT"
100.0            1       20,001           1           2   20001584 main() (calls f1, f5)
0,0,0 5 "main() (calls f1, f5)" incl 20001584 100.00 GROUP="TAU_DEFAULT"
100.0            1       20,001           1           2   20001584 main() (calls f1, f5)
0,0,0 6 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  " incl 15000638 75.00 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000638 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  
0,0,0 0 "f1() (sleeps 1 sec, calls f2, f4)" incl 15000638 75.00 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000638 f1() (sleeps 1 sec, calls f2, f4)
0,0,0 1 "f2() (sleeps 2 sec, calls f3)" incl 10000314 50.00 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000157 f2() (sleeps 2 sec, calls f3)
0,0,0 9 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  " incl 9000213 45.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000213 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  
0,0,0 3 "f4() (sleeps 4 sec, calls f2)" incl 9000213 45.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000213 f4() (sleeps 4 sec, calls f2)
0,0,0 2 "f3() (sleeps 3 sec)" incl 5999972 30.00 GROUP="TAU_USER"
 30.0        5,999        5,999           2           0    2999986 f3() (sleeps 3 sec)
0,0,0 10 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " incl 5000167 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000167 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
0,0,0 7 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  " incl 5000147 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000147 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  
0,0,0 12 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " incl 4999923 25.00 GROUP="TAU_CALLPATH"
 25.0        4,999        4,999           1           0    4999923 main() (calls f1, f5) => f5() (sleeps 5 sec)  
0,0,0 4 "f5() (sleeps 5 sec)" incl 4999923 25.00 GROUP="TAU_USER"
 25.0        4,999        4,999           1           0    4999923 f5() (sleeps 5 sec)
0,0,0 11 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 2999988 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999988 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
0,0,0 8 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 2999984 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999984 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
t 2 "f3() (sleeps 3 sec)" excl 5999972 30.00 GROUP="TAU_USER"
 30.0        5,999        5,999           2           0    2999986 f3() (sleeps 3 sec)
t 4 "f5() (sleeps 5 sec)" excl 4999923 25.00 GROUP="TAU_USER"
 25.0        4,999        4,999           1           0    4999923 f5() (sleeps 5 sec)
t 12 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " excl 4999923 25.00 GROUP="TAU_CALLPATH"
 25.0        4,999        4,999           1           0    4999923 main() (calls f1, f5) => f5() (sleeps 5 sec)  
t 1 "f2() (sleeps 2 sec, calls f3)" excl 4000342 20.00 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000157 f2() (sleeps 2 sec, calls f3)
t 9 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  " excl 4000046 20.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000213 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  
t 3 "f4() (sleeps 4 sec, calls f2)" excl 4000046 20.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000213 f4() (sleeps 4 sec, calls f2)
t 11 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 2999988 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999988 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
t 8 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 2999984 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999984 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
t 10 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " excl 2000179 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000167 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
t 7 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  " excl 2000163 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000147 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  
t 6 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  " excl 1000278 5.00 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000638 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  
t 0 "f1() (sleeps 1 sec, calls f2, f4)" excl 1000278 5.00 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000638 f1() (sleeps 1 sec, calls f2, f4)
t 5 "main() (calls f1, f5)" excl 1023 0.01 GROUP="TAU_DEFAULT"
100.0            1       20,001           1           2   20001584 main() (calls f1, f5)
t 5 "main() (calls f1, f5)" incl 20001584 100.00 GROUP="TAU_DEFAULT"
100.0            1       20,001           1           2   20001584 main() (calls f1, f5)
t 6 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  " incl 15000638 75.00 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000638 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  
t 0 "f1() (sleeps 1 sec, calls f2, f4)" incl 15000638 75.00 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000638 f1() (sleeps 1 sec, calls f2, f4)
t 1 "f2() (sleeps 2 sec, calls f3)" incl 10000314 50.00 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000157 f2() (sleeps 2 sec, calls f3)
t 9 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  " incl 9000213 45.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000213 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  
t 3 "f4() (sleeps 4 sec, calls f2)" incl 9000213 45.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000213 f4() (sleeps 4 sec, calls f2)
t 2 "f3() (sleeps 3 sec)" incl 5999972 30.00 GROUP="TAU_USER"
 30.0        5,999        5,999           2           0    2999986 f3() (sleeps 3 sec)
t 10 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " incl 5000167 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000167 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
t 7 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  " incl 5000147 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000147 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  
t 12 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " incl 4999923 25.00 GROUP="TAU_CALLPATH"
 25.0        4,999        4,999           1           0    4999923 main() (calls f1, f5) => f5() (sleeps 5 sec)  
t 4 "f5() (sleeps 5 sec)" incl 4999923 25.00 GROUP="TAU_USER"
 25.0        4,999        4,999           1           0    4999923 f5() (sleeps 5 sec)
t 11 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 2999988 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999988 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
t 8 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 2999984 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999984 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
m 2 "f3() (sleeps 3 sec)" excl 5999972 30.00 GROUP="TAU_USER"
 30.0        5,999        5,999           2           0    2999986 f3() (sleeps 3 sec)
m 12 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " excl 4999923 25.00 GROUP="TAU_CALLPATH"
 25.0        4,999        4,999           1           0    4999923 main() (calls f1, f5) => f5() (sleeps 5 sec)  
m 4 "f5() (sleeps 5 sec)" excl 4999923 25.00 GROUP="TAU_USER"
 25.0        4,999        4,999           1           0    4999923 f5() (sleeps 5 sec)
m 1 "f2() (sleeps 2 sec, calls f3)" excl 4000342 20.00 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000157 f2() (sleeps 2 sec, calls f3)
m 9 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  " excl 4000046 20.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000213 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  
m 3 "f4() (sleeps 4 sec, calls f2)" excl 4000046 20.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000213 f4() (sleeps 4 sec, calls f2)
m 11 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 2999988 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999988 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
m 8 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " excl 2999984 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999984 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
m 10 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " excl 2000179 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000167 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
m 7 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  " excl 2000163 10.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000147 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  
m 0 "f1() (sleeps 1 sec, calls f2, f4)" excl 1000278 5.00 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000638 f1() (sleeps 1 sec, calls f2, f4)
m 6 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  " excl 1000278 5.00 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000638 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  
m 5 "main() (calls f1, f5)" excl 1023 0.01 GROUP="TAU_DEFAULT"
100.0            1       20,001           1           2   20001584 main() (calls f1, f5)
m 5 "main() (calls f1, f5)" incl 20001584 100.00 GROUP="TAU_DEFAULT"
100.0            1       20,001           1           2   20001584 main() (calls f1, f5)
m 0 "f1() (sleeps 1 sec, calls f2, f4)" incl 15000638 75.00 GROUP="TAU_USER"
 75.0        1,000       15,000           1           2   15000638 f1() (sleeps 1 sec, calls f2, f4)
m 6 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  " incl 15000638 75.00 GROUP="TAU_CALLPATH"
 75.0        1,000       15,000           1           2   15000638 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4)  
m 1 "f2() (sleeps 2 sec, calls f3)" incl 10000314 50.00 GROUP="TAU_USER"
 50.0        4,000       10,000           2           2    5000157 f2() (sleeps 2 sec, calls f3)
m 9 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  " incl 9000213 45.00 GROUP="TAU_CALLPATH"
 45.0        4,000        9,000           1           1    9000213 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2)  
m 3 "f4() (sleeps 4 sec, calls f2)" incl 9000213 45.00 GROUP="TAU_USER"
 45.0        4,000        9,000           1           1    9000213 f4() (sleeps 4 sec, calls f2)
m 2 "f3() (sleeps 3 sec)" incl 5999972 30.00 GROUP="TAU_USER"
 30.0        5,999        5,999           2           0    2999986 f3() (sleeps 3 sec)
m 10 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  " incl 5000167 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000167 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3)  
m 7 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  " incl 5000147 25.00 GROUP="TAU_CALLPATH"
 25.0        2,000        5,000           1           1    5000147 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3)  
m 4 "f5() (sleeps 5 sec)" incl 4999923 25.00 GROUP="TAU_USER"
 25.0        4,999        4,999           1           0    4999923 f5() (sleeps 5 sec)
m 12 "main() (calls f1, f5) => f5() (sleeps 5 sec)  " incl 4999923 25.00 GROUP="TAU_CALLPATH"
 25.0        4,999        4,999           1           0    4999923 main() (calls f1, f5) => f5() (sleeps 5 sec)  
m 11 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 2999988 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999988 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f4() (sleeps 4 sec, calls f2) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
m 8 "main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  " incl 2999984 15.00 GROUP="TAU_CALLPATH"
 15.0        2,999        2,999           1           0    2999984 main() (calls f1, f5) => f1() (sleeps 1 sec, calls f2, f4) => f2() (sleeps 2 sec, calls f3) => f3() (sleeps 3 sec)  
