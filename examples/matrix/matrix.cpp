/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/*--------------------------------------------------------------------
|
|  n-processor matrix muliplication
|
|  Author:
|    Mike Kaufman
|    University of Oregon, Dept. of Computer Science
|    mikek@cs.uoregon.edu
|  
|  Purpose:  This program is intended to illustrate the use of the TAU 
|            Parallel Profiling Package.  As an example, we multiply
|            four sets of 20x20 matrices.  Each set is instantiated with
|            a different type; ints, longs, floats and doubles are used 
|            to illustrate the TAU Parallel Profiling Package's ability
|            to differentiate between various type instantiations of 
|            templated classes.  Communication between processors is 
|            performed with the MPI library.  
|
*/  


#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <Profile/Profiler.h>
#include <string.h>

#define TRUE  1;
#define FALSE 0;
// mpi master to slave messages
const int SLAVE_TERMINATE        = 0;
const int SLAVE_EXPECT_BROADCAST = 1;
const int SLAVE_EXPECT_MESSAGE   = 2;


/*--------------------------------------------------------------------
|  
| Matrix_MPI
| ==========
|
| This is a templated matrix class.  At this time, only the 
| * (multiplication) operator and = (assignment) operator are overloaded.
| Any type which Matrix_MPI is instantiated with must have the following
| operators defined:
|   
|     *        (multiplication)
|     =        (assignment)
|     <<       (stream insertion)
|     T(int)   (int to <class T> conversion constructor)
|
| MPI must be started before a Matrix_MPI object is created and MPI must
| be finalized after all Matrix_MPI objects have been created.
|
| The matrix is stored on only the master MPI thread (assumed to be MPI 
| rank 0).  During the multiplication, the master thread will dispatch 
| vectors to the available slave threads and wait for the slaves to return
| resultant values.
| 
| the print() method will output its matrix to stdout.
*/
template<class T> 
class Matrix_MPI
{
  public:
    Matrix_MPI();
    Matrix_MPI(Matrix_MPI&);
    Matrix_MPI(int m, int n, int rank, int np);
    ~Matrix_MPI();
    Matrix_MPI operator*(Matrix_MPI& b);
    Matrix_MPI operator=(Matrix_MPI arg);
    void   print();
    T** a;            // the nrowsXncols matrix of type T
    int nrows, ncols;
    int nprocs;
    int mpiRank;
    int master;



};
/*------------------------------------------------------------------*/


/*--------------------------------------------------------------------
|
|  Matrix_MPI::Matrix_MPI()
|  ========================
|
|  no-arg constructor.  Creates a Matrix_MPI object but does not
|  allocate any space for the matrix
*/
template<class T>
Matrix_MPI<T>::Matrix_MPI()
{
  a = NULL;
}
/*------------------------------------------------------------------*/


/*--------------------------------------------------------------------
|
|  Matrix_MPI::Matrix_MPI(Matrix_MPI)
|  ==================================
|  
|  copy constructor.  Allocate unique space for the matrix, 
|  value-to-value copy the other stuff.
|
*/
template<class T>
Matrix_MPI<T>::Matrix_MPI(Matrix_MPI<T>& arg)
{

  int i, j;

  // TAU profiling
  TAU_TYPE_STRING(str, CT(arg) + " (" + CT(arg)+")");
  TAU_PROFILE("Matrix_MPI::Matrix_MPI()", str, TAU_DEFAULT);

  // just copy these members
  nrows   =  arg.nrows;
  ncols   =  arg.ncols;
  master  = arg.master;
  nprocs  = arg.nprocs;
  mpiRank = arg.mpiRank;
  a       = NULL;

  if (mpiRank == master)
  {
    // alocate space for matrix data
    a = new T*[nrows]; 
    for (i=0; i<nrows; i++)
      a[i] = new T[ncols];
    
    // copy data from old matrix to new matrix
    for (i=0; i<nrows; i++)
      for (j=0; j<ncols; j++)
        a[i][j] = arg.a[i][j];
  }
}
/*------------------------------------------------------------------*/


/*--------------------------------------------------------------------
|
|  Matrix_MPI::Matrix_MPI(int m, int n, int rank, int np)
|  =====================
|
|  constructor.  Creates a Matrix_MPI object and allocates an mXn
|  2-D array of type T for the matrix.
|
*/
template<class T> 
Matrix_MPI<T>::Matrix_MPI(int m, int n, int rank, int np)
{ 
  // m is number of rows of matrix
  // n is number of columns of matrix
  // rank is the MPI processor rank/ID
  // np is the number of processors that this program is being run on

  int i, j;

  
  // TAU profiling
  TAU_TYPE_STRING(str, CT(*this) + " (int, int, int, int)"); 
  TAU_PROFILE("Matrix_MPI::Matrix_MPI()", str, TAU_DEFAULT);
 
  // set mpiRank
  mpiRank = rank;
  master  = 0;
  // set rows and columns
  nrows   = m;
  ncols   = n; 
  nprocs  = np;
  a       = NULL;

  // allocate space for matrix *only* if this is master thread 
  if (mpiRank == master)
  {
    // try allocating space in one big chunk
      a = new T*[nrows]; 
      for (i=0; i<nrows; i++)
        a[i] = new T[ncols];
   
      // init every value to one
      for (i=0; i<nrows; i++)
        for (j=0; j<ncols; j++)
        {
          a[i][j] = T(1); 
        }
  }   

}


/*--------------------------------------------------------------------
|
|  Matrix_MPI::~Matrix_MPI()
|  =========================
|
|  Matrix_MPI destructor
|
*/
template<class T> 
Matrix_MPI<T>::~Matrix_MPI()
{
  // reclaim matrix's memory
  int i;

  // TAU profiling
  TAU_TYPE_STRING(str, CT(*this) + " void (void)"); 
  TAU_PROFILE("Matrix_MPI::~Matrix_MPI()", str, TAU_DEFAULT);

  if (mpiRank == master)
  {  
    for(i=0; i<nrows; i++)
      delete[] a[i];
    delete[] a;
  }  
}
/*------------------------------------------------------------------*/


/*--------------------------------------------------------------------
|
|  Matrix_MPI::print()
|  ===================
|
|  output matrix
|
*/
template<class T>
void Matrix_MPI<T>::print()
{
   
  // TAU profiling
  TAU_TYPE_STRING(str, CT(*this) + " void (void)"); 
  TAU_PROFILE("Matrix_MPI::print()", str, TAU_DEFAULT);
  
  int i,j;  
  // output matrix only if we are master process
  if (mpiRank == master)  
   for (i=0; i<nrows; i++)
   {
     for (j=0; j<ncols; j++)
       cout << a[i][j] <<  "   ";
     cout << endl;
   }	
}
/*------------------------------------------------------------------*/


/*--------------------------------------------------------------------
|
|  Matrix_MPI::operator = 
|  ======================
|
|  Assignment operator
|
*/
template<class T>
Matrix_MPI<T> Matrix_MPI<T>::operator=(Matrix_MPI<T> arg)
{
  // assign matrix b to this matrix
  int i, j;  

  // TAU profiling
  TAU_TYPE_STRING(str, CT(arg) + " (" + CT(arg) + ")");
  TAU_PROFILE("Matrix_MPI::operator=()", str, TAU_DEFAULT);

  // make sure rows and columns jive
  if ((nrows != arg.nrows) || (ncols != arg.ncols))
  {
   cerr << "Unable to assign an " << arg.nrows << "X" << arg.ncols << "matrix";
   cerr << " to a " << nrows << "X" << ncols << endl;           
  }
  else 
    if (mpiRank == master)
    {  
      // assign matrix
      for (i=0; i<nrows; i++)
        for (j=0; j<ncols; j++)    
          a[i][j] = arg.a[i][j];
    }  

  return *this;

}
/*------------------------------------------------------------------*/


/*--------------------------------------------------------------------
|
|  Matrix_MPI::operator *
|  ======================
|
|  multiplication operator.  The MPI master thread (always thread 0) 
|  distributes vectors to the slaves and then waits for the slaves 
|  to return the product.  The slaves, consequently, compute the 
|  product and return it to the master.  When master has distributed 
|  all vectors and received all results, it sends slaves a terminate 
|  message and slaves will terminate their loop.
|  
*/
template<class T>
Matrix_MPI<T> Matrix_MPI<T>::operator*(Matrix_MPI<T>& arg)
{
  
  int currCol;     // current column of b that we are multiplying
  int currRow;     // current row of a that we are multiplying
  int procs;       // count of processors that have been given data
  int sender;      // MPI process ID of last processor to return a result
  int resultCol;   // column # of result returned to master by slave
  int procsFilled; // true if all processors have been assigned at least one
                   // task of multiplying two vectors   
  int terminate;   // false until master sends SLAVE_TERMINATE msg to slave
  int i, j, recvcount;
  T ans;
  T* vctr1 = new T[ncols];  // a row of this that is bcast'ed to slave
  T* vctr2 = new T[ncols];  // a column of arg sent to slave & returned to 
                            //     master
  MPI_Status stat;
  Matrix_MPI<T> rm(nrows, arg.ncols,mpiRank, nprocs);  // resultant matrix 
  
  
  // TAU profiling
  TAU_TYPE_STRING(str,  CT(arg) +" (" + CT(arg) + ")"); 
  TAU_PROFILE("Matrix_MPI::operator*()", str, TAU_DEFAULT);

  // a timer for the multiplication one row in left matrix by
  //    each column in right matrix
  TAU_PROFILE_TIMER(rowMatrixTimer, "row-by-matrix_Timer", str, TAU_USER);
  // a timer for vector by vector muliplication
  TAU_PROFILE_TIMER(vectorMultTimer, "vector-by-vector_Timer", str, TAU_USER);

  if (mpiRank == master)
  {     
    // first verify matrix multiplaction is defined
    if (this->ncols != arg.nrows)  
    {
      cerr << "can not multiply an "<< nrows << "X" << ncols;
      cerr << "matrix with a ";
      cerr << arg.nrows << "X" << arg.ncols << "matrix" << endl;
      Matrix_MPI<T> temp;
      
      return temp;
    }

    // now multiply the matrices
    // for each row in left matrix
    for (currRow=0; currRow<nrows; currRow++)
    {  
      // start rowTimer
      TAU_PROFILE_START(rowMatrixTimer);

      // tell slaves to expect broadcast
      for (i=1; i<nprocs; i++)
      {
	TAU_TRACE_SENDMSG(SLAVE_EXPECT_BROADCAST, i, 0);
        MPI_Send(0, 0, MPI_INT, i, SLAVE_EXPECT_BROADCAST, MPI_COMM_WORLD);
      }
      // broadcast row of left matrix to slaves
      for(i=1; i<nprocs; i++)
	TAU_TRACE_SENDMSG(master, i, ncols*sizeof(T));
      MPI_Bcast(a[currRow], ncols * sizeof(T), MPI_BYTE, master, 
		MPI_COMM_WORLD);    
      
      currCol     = 0;
      procs       = 0;
      sender      = 0;
      procsFilled = FALSE;


      while (currCol < arg.ncols)
      {
        procs++;
        if (!procsFilled)      
          sender = procs;
        // tell slave to expect message
	TAU_TRACE_SENDMSG(SLAVE_EXPECT_MESSAGE, sender, 0);
        MPI_Send(0, 0, MPI_INT, sender, SLAVE_EXPECT_MESSAGE, 
		 MPI_COMM_WORLD);


        // send individual columns of right matrix to individual 
        //      processors
        for (j=0; j<ncols; j++)
          vctr2[j] = arg.a[j][currCol];
	TAU_TRACE_SENDMSG(currCol, sender, ncols * sizeof(T));
        MPI_Send(vctr2, ncols * sizeof(T), MPI_BYTE, sender, currCol, 
		 MPI_COMM_WORLD);
	
        currCol++;
        
        if (procs == (nprocs-1))
	{

          // we have sent something to all of our slaves so wait for result
          MPI_Recv(&ans, sizeof(T), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG,
		   MPI_COMM_WORLD, &stat);
	  MPI_Get_count(&stat, MPI_BYTE, &recvcount);
	  TAU_TRACE_RECVMSG(stat.MPI_TAG, stat.MPI_SOURCE, recvcount);
          procsFilled = TRUE;
          procs--;
          resultCol = stat.MPI_TAG;
          sender    = stat.MPI_SOURCE;    // retain the free processor
	  rm.a[currRow][resultCol] = ans;
        }
      }
      
      // get remaining answers that are out there
      for (j=0; j<procs; j++)
      {
        MPI_Recv(&ans, sizeof(T), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG,
		 MPI_COMM_WORLD, &stat);
	MPI_Get_count(&stat, MPI_BYTE, &recvcount);
	TAU_TRACE_RECVMSG(stat.MPI_TAG, stat.MPI_SOURCE, recvcount);
        resultCol = stat.MPI_TAG;
        rm.a[currRow][resultCol] = ans;

      }
   
      // stop row timer
      TAU_PROFILE_STOP(rowMatrixTimer);  
    }
    
    // master is finished so broadcast termination message
    for (i=1; i<nprocs; i++)
    {
       TAU_TRACE_SENDMSG(SLAVE_TERMINATE, i, 0);
       MPI_Send(0, 0, MPI_INT, i, SLAVE_TERMINATE, MPI_COMM_WORLD);
    }

  }

  else
  {
    // slave code    
    terminate = FALSE;
   

    while (!terminate)
    {
      // receive primary message
      MPI_Recv(0, 0, MPI_INT, master, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
      MPI_Get_count(&stat, MPI_BYTE, &recvcount);
      TAU_TRACE_RECVMSG(stat.MPI_TAG, stat.MPI_SOURCE, recvcount );
      
      switch(stat.MPI_TAG)
      {  
        case SLAVE_TERMINATE:
          terminate = TRUE;
          break;

        case SLAVE_EXPECT_BROADCAST:
          // receive b-vector from MPI_Broadcast
          MPI_Bcast(vctr1, ncols * sizeof(T), MPI_BYTE, master,
		    MPI_COMM_WORLD);
          break;

        case SLAVE_EXPECT_MESSAGE:

          // start vectorMult timer
          TAU_PROFILE_START(vectorMultTimer);

          // receive a unique vector to multiply with b
          MPI_Recv(vctr2, ncols * sizeof(T), MPI_BYTE, master, MPI_ANY_TAG,
		   MPI_COMM_WORLD,&stat);
	  MPI_Get_count(&stat, MPI_BYTE, &recvcount);
	  TAU_TRACE_RECVMSG(stat.MPI_TAG, stat.MPI_SOURCE, recvcount);
          // now compute the vector product vctr1 * vctr2
          ans = 0;
          for (i=0; i<ncols; i++)
            ans = ans + (vctr1[i] * vctr2[i]);
          // now send answer back to master
	  TAU_TRACE_SENDMSG(stat.MPI_TAG, master, sizeof(T));
          MPI_Send(&ans, sizeof(T), MPI_BYTE, master, stat.MPI_TAG, 
		   MPI_COMM_WORLD);
      
          // stop vectorMultTimer
          TAU_PROFILE_STOP(vectorMultTimer);
	  break;

        default:
          cerr << "invalid message passed to slave" << endl;
          break;
      }
    }

  }

  // clean up 
  delete[] vctr1;
  delete[] vctr2;
  
  return rm;
  
}
/*------------------------------------------------------------------*/



void main(int argc, char** argv)
{
  // MPI vars
  int np, namelen;
  int myid;                                // MPI processor id
  char proc_name[MPI_MAX_PROCESSOR_NAME];

  
  // MPI stuff
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(proc_name, &namelen);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  // TAU stuff
  TAU_PROFILE_SET_NODE(myid);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE("main()", "void (int, char**)", TAU_DEFAULT);
  
  // make sure we have at least 2 processors
  if (np < 2)
  { 
    cerr << endl << endl << endl;
    cerr << "ERROR:  This program must be run with a minimum of 2 processors!";
    cerr << endl<< endl << endl;
    MPI_Finalize();
    exit(1);
  }
  
  // mulitply some matrices
 
  Matrix_MPI<int> a(20, 20, myid, np);
  Matrix_MPI<int> b(20, 20, myid, np);
  Matrix_MPI<int> c(20, 20, myid, np);
  c = a * b;
   
  Matrix_MPI<long> d(20, 20, myid, np);
  Matrix_MPI<long> e(20, 20, myid, np);
  Matrix_MPI<long> f(20, 20, myid, np);
  f = d*e;

 
  Matrix_MPI<float> g(20, 20, myid, np);
  Matrix_MPI<float> h(20, 20, myid, np);
  Matrix_MPI<float> i(20, 20, myid,np);
  i = g * h; 
 
  Matrix_MPI<double> j(20, 20, myid, np);
  Matrix_MPI<double> k(20, 20, myid, np);
  Matrix_MPI<double> l(20, 20, myid,np);
  l = j * k;

  if (myid == 0)
    cout << "Multiplied matrices of types int, long, float and double" << endl;
    
  MPI_Finalize();


}






