
void initialize(float* a, float* b, float* c, int M)
{
//        float* a = (float*)malloc(matsize);
//        float* b = (float*)malloc(matsize);
//        float* c = (float*)malloc(matsize);

        //initalize matrices
	int i,j;
#pragma omp parallel private(i,j)
#pragma omp parallel for
        for (i=0; i<M; i++) {
                for (j=0; j<M; j++) {
                        //a[i*m+j] = i;
                        //b[i*m+j] = i;
                        a[i*M+j] = i-j*2 + i-j+1 + 1;
                        b[i*M+j] = i-j*2 + i-j+1 + 1;
                        c[i*M+j] = 0;
                        //std::cout << a[i*m+j] << ", ";
                }
                //std::cout << std::endl;
        }

}
