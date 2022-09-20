  #include <mpi.h>

  int main(int argc, char **argv)
  {
    int ret, req, provided;
    req = 0;
    ret = PMPI_Init_thread(&argc, &argv, req, &provided);
    return ret;
  }

