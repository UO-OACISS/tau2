#include <mpi.h>

#include <iostream>
#include <map>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
  char name[64];
  int rank;
  int size;

  map<string,int> hostHash;
  map<string,int>::iterator it;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  gethostname(name, sizeof(name));

  if (rank == 0) {
    string *hostName = new string(name);
    hostHash[*hostName] = 1;
    MPI_Status status;
    for (int i=1; i<size; i++) {
      MPI_Recv(name, 64, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      hostName = new string(name);
      if (hostHash.count(*hostName) > 0) {
	hostHash[*hostName] = hostHash[*hostName] + 1;
      } else {
	hostHash[*hostName] = 1;
      }
    }
  } else {
    MPI_Send(name, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    FILE *outfile;
    outfile = fopen("allhosts.txt","w");
    for (it=hostHash.begin(); it!=hostHash.end(); it++) {
      fprintf(outfile,"%d:%s\n",(*it).second,(*it).first.c_str());
    }
    //     fprintf(outfile,"-1:end\n");
    fclose(outfile);
  }

  MPI_Finalize();
}
