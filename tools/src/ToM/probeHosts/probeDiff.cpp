#include <mpi.h>

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>

#include <map>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
  int node;
  char name[64];
  int rank;
  int size;

  map<string,int> hostHash;
  map<string,int>::iterator it;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  FILE *fullHostFile;
  char *outdirName;
  outdirName = getenv("PROFILEDIR");
  if (outdirName == NULL) {
    outdirName = (char *)malloc((strlen(".")+1)*sizeof(char));
    strcpy(outdirName,".");
  }
  char infileString[512];
  sprintf(infileString,"%s/allhosts.txt",outdirName);
  fullHostFile = fopen(infileString, "r");

  // Building hash table counts for hosts in a full job reservation
  if (rank == 0) {
    int numHosts = 0;
    char hostName[64];
    string *hostString;
    while (fscanf(fullHostFile, "%d:%s\n", &numHosts, &hostName) != EOF) {
      // break condition
      /*
	if (numHosts == -1); {
	printf("Numhosts is -1 and name is [%s]\n", hostName);
	break;
	}
      */
      hostString = new string(hostName);
      hostHash[*hostString] = numHosts;
    }
    fclose(fullHostFile);
  }

  /* DEBUG
  for (it=hostHash.begin(); it!=hostHash.end(); it++) {
    printf("%d:%s\n",(*it).second,(*it).first.c_str());
  }
  */

  // Subtracting from the full hash table what is intended to be used
  //   in the user MPI application. What remains gets to be used for
  //   the MRNet topology tree.

  gethostname(name, sizeof(name));
  if (rank == 0) {
    string *hostName = new string(name);
    assert(hostHash[*hostName] > 0);
    hostHash[*hostName] = hostHash[*hostName] - 1;
    MPI_Status status;
    for (int i=1; i<size; i++) {
      MPI_Recv(name, 64, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      hostName = new string(name);
      string *hostName = new string(name);
      assert(hostHash[*hostName] > 0);
      hostHash[*hostName] = hostHash[*hostName] - 1;
    }
  } else {
    MPI_Send(name, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    FILE *outfile;
    char outfileString[512];
    sprintf(outfileString,"%s/mrnethosts.txt",outdirName);
    outfile = fopen(outfileString,"w");
    for (it=hostHash.begin(); it!=hostHash.end(); it++) {
      // remove all references to hosts completely consumed by the MPI
      //   process space.
      if ((*it).second > 0) {
	fprintf(outfile,"%s:%d\n",(*it).first.c_str(),(*it).second);
      }
    }
    //     fprintf(outfile,"-1:end\n");
    fclose(outfile);
  }

  MPI_Finalize();
}
