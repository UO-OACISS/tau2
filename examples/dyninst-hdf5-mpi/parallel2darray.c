#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "hdf5.h"


#define MAXFILENAME 128
typedef struct rundata {
    int globalnx, globalny;
    int localnx, localny;
    int npx, npy, nprocs, rank;
    int myx, myy;
    char filename[MAXFILENAME]; 
} rundata_t;


void writehdf5file(rundata_t rundata, double **dens, double ***vel) {
    /* identifiers */
    hid_t file_id, arr_group_id, dens_dataset_id, vel_dataset_id;
    hid_t dens_dataspace_id, vel_dataspace_id;
    hid_t loc_dens_dataspace_id, loc_vel_dataspace_id;
    hid_t globaldensspace,globalvelspace;
    hid_t dist_id;
    hid_t fap_id;

    /* sizes */
    hsize_t densdims[2], veldims[3];
    hsize_t locdensdims[2], locveldims[3];

    /* status */
    herr_t status;

    /* MPI-IO hints for performance */
    MPI_Info info;

    /* parameters of the hyperslab */
    hsize_t counts[3];
    hsize_t strides[3];
    hsize_t offsets[3];
    hsize_t blocks[3];

    /* set the MPI-IO hints for better performance on GPFS */
    MPI_Info_create(&info);
    MPI_Info_set(info,"IBM_largeblock_io","true");

    /* Set up the parallel environment for file access*/
    fap_id = H5Pcreate(H5P_FILE_ACCESS);
    /* Include the file access property with IBM hint */
    H5Pset_fapl_mpio(fap_id, MPI_COMM_WORLD, info);

    /* Set up the parallel environment */
    dist_id = H5Pcreate(H5P_DATASET_XFER);
    /* we'll be writing collectively */
    H5Pset_dxpl_mpio(dist_id, H5FD_MPIO_COLLECTIVE);

    /* Create a new file - truncate anything existing, use default properties */
    file_id = H5Fcreate(rundata.filename, H5F_ACC_TRUNC, H5P_DEFAULT, fap_id);

    /* HDF5 routines generally return a negative number on failure.  
     * Should check return values! */
    if (file_id < 0) {
        fprintf(stderr,"Could not open file %s\n", rundata.filename);
        return;
    }

    /* Create a new group within the new file */
    arr_group_id = H5Gcreate(file_id,"/ArrayData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Give this group an attribute listing the time of calculation */
    {
        hid_t attr_id,attr_sp_id;
        struct tm *t;
        time_t now;
        int yyyymm;
        now = time(NULL);
        t = localtime(&now);
        yyyymm = (1900+t->tm_year)*100+t->tm_mon;     

        attr_sp_id = H5Screate(H5S_SCALAR);
        attr_id = H5Acreate(arr_group_id, "Calculated on (YYYYMM)", H5T_STD_U32LE, attr_sp_id, H5P_DEFAULT, H5P_DEFAULT);
        printf("yymm = %d\n",yyyymm);
        H5Awrite(attr_id, H5T_NATIVE_INT, &yyyymm);
        H5Aclose(attr_id);
        H5Sclose(attr_sp_id);
    }

    /* Create the data space for the two global datasets. */
    densdims[0] = rundata.globalnx; densdims[1] = rundata.globalny;
    veldims[0] = 2; veldims[1] = rundata.globalnx; veldims[2] = rundata.globalny;
    
    dens_dataspace_id = H5Screate_simple(2, densdims, NULL);
    vel_dataspace_id  = H5Screate_simple(3, veldims,  NULL);

    /* Create the datasets within the file. 
     * H5T_IEEE_F64LE is a standard (IEEE) double precision (64 bit) floating (F) data type
     * and will work on any machine.  H5T_NATIVE_DOUBLE would work too, but would give
     * different results on GPC and TCS */

    dens_dataset_id = H5Dcreate(file_id, "/ArrayData/dens", H5T_IEEE_F64LE, 
                                dens_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    vel_dataset_id  = H5Dcreate(file_id, "/ArrayData/vel",  H5T_IEEE_F64LE, 
                                vel_dataspace_id,  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Now create the data space for our sub-regions.   These are the data spaces
     * of our actual local data in memory. */
    locdensdims[0] = rundata.localnx; locdensdims[1] = rundata.localny;
    locveldims[0] = 2; locveldims[1] = rundata.localnx; locveldims[2] = rundata.localny;
    
    loc_dens_dataspace_id = H5Screate_simple(2, locdensdims, NULL);
    loc_vel_dataspace_id  = H5Screate_simple(3, locveldims,  NULL);

    /*
     *
     * Now we have to figure out the `hyperslab' within the global
     * data that corresponds to our local data.
     *
     * Hyperslabs are described by an array of counts, strides, offsets,
     * and block sizes.
     *
     *       |-offx--|
     *       +-------|----|-------+   -+-
     *       |                    |    |
     *       |                    |   offy
     *       |                    |    |
     *       -       +----+       -   -+-
     *       |       |    |       |    |
     *       |       |    |       |  localny
     *       |       |    |       |    |
     *       -       +----+       -   -+-
     *       |                    |
     *       |                    |
     *       +-------|----|-------+
     *               localnx
     *
     *  In this case the blocksizes are (localnx,localny) and the offsets are 
     *  (offx,offy) = ((myx)/nxp*globalnx, (myy/nyp)*globalny)
     */

    offsets[0] = (rundata.globalnx/rundata.npx)*rundata.myx;
    offsets[1] = (rundata.globalny/rundata.npy)*rundata.myy;
    blocks[0]  = rundata.localnx;
    blocks[1]  = rundata.localny;
    strides[0] = strides[1] = 1;
    counts[0] = counts[1] = 1;

    /* select this subset of the density variable's space in the file */
    globaldensspace = H5Dget_space(dens_dataset_id);
    H5Sselect_hyperslab(globaldensspace,H5S_SELECT_SET, offsets, strides, counts, blocks);

    /* For the velocities, it's the same thing but there's a count of two,
     * (one for each velocity component) */

    offsets[1] = (rundata.globalnx/rundata.npx)*rundata.myx;
    offsets[2] = (rundata.globalny/rundata.npy)*rundata.myy;
    blocks[1]  = rundata.localnx;
    blocks[2]  = rundata.localny;
    strides[0] = strides[1] = strides[2] = 1;
    counts[0] = 2; counts[1] = counts[2] = 1;
    offsets[0] = 0;
    blocks[0] = 1;

    globalvelspace = H5Dget_space(vel_dataset_id);
    H5Sselect_hyperslab(globalvelspace,H5S_SELECT_SET, offsets, strides, counts, blocks);

    /* Write the data.  We're writing it from memory, where it is saved 
     * in NATIVE_DOUBLE format */
    status = H5Dwrite(dens_dataset_id, H5T_NATIVE_DOUBLE, loc_dens_dataspace_id, globaldensspace, dist_id, &(dens[0][0]));
    status = H5Dwrite(vel_dataset_id,  H5T_NATIVE_DOUBLE, loc_vel_dataspace_id, globalvelspace, dist_id, &(vel[0][0][0]));

    /* We'll create another group for related info and put some things in there */

    {
        hid_t other_group_id;
        hid_t timestep_id, timestep_space;
        hid_t comptime_id, comptime_space;
        hid_t author_id, author_space, author_type;
        char *authorname="Jonathan Dursi";
        int timestep=13;
        float comptime=81.773;

        /* create group */
        other_group_id = H5Gcreate(file_id,"/OtherStuff", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* scalar space, data for integer timestep */
        timestep_space = H5Screate(H5S_SCALAR);
        timestep_id = H5Dcreate(other_group_id, "Timestep", H5T_STD_U32LE, 
                                timestep_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(timestep_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &timestep);
        H5Dclose(timestep_id);
        H5Sclose(timestep_space);
   
        /* scalar space, data for floating compute time */
        comptime_space = H5Screate(H5S_SCALAR);
        comptime_id = H5Dcreate(other_group_id, "Compute Time", H5T_IEEE_F32LE, 
                                comptime_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(comptime_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &comptime);
        H5Dclose(comptime_id);
        H5Sclose(comptime_space);
   
        /* scalar space, data for author name */
        author_space = H5Screate(H5S_SCALAR);
        author_type  = H5Tcopy(H5T_C_S1);   /* copy the character type.. */
        status = H5Tset_size (author_type, strlen(authorname));  /* and make it longer */
        author_id = H5Dcreate(other_group_id, "Simulator Name", author_type, author_space, 
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
        status = H5Dwrite(author_id, author_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, authorname);
        H5Dclose(author_id);
        H5Sclose(author_space);
        H5Tclose(author_type);

        H5Gclose(other_group_id);
    } 


    /* End access to groups & data sets and release resources used by them */
    status = H5Sclose(dens_dataspace_id);
    status = H5Dclose(dens_dataset_id);
    status = H5Sclose(vel_dataspace_id);
    status = H5Dclose(vel_dataset_id);
    status = H5Gclose(arr_group_id);
    status = H5Pclose(fap_id);
    status = H5Pclose(dist_id);
    
    /* Close the file */
    status = H5Fclose(file_id);
    return;
}


int get_options(int argc, char **argv, rundata_t *rundata);
double **array2d(int nx, int ny);
double ***array3d(int nd, int nx, int ny);
void freearray2d(double **d);
void freearray3d(double ***d);
void fillarray2d(double **d, rundata_t *r);
void fillarray3d(double ***v, rundata_t *r);
void printarray2d(double **d, int nx, int ny);
void nearsquare(int nprocs, int *npx, int *npy);


int main(int argc, char **argv) {
    double **locdens;
    double ***locvel;
    rundata_t rundata;
    int ierr;
    int rank, size;


    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    /*  
     * set default values for parameters, then check what the user says
     * with the options 
     */

    /* choose sensible defaults */
    nearsquare(size,&(rundata.npx),&(rundata.npy));
    rundata.globalnx = 100;
    rundata.globalny = 100;
    rundata.nprocs = size;
    rundata.rank = rank;
    rundata.localnx = (rundata.globalnx / rundata.npx);
    rundata.localny = (rundata.globalny / rundata.npy);
    strcpy(rundata.filename,"paralleldata.h5");

    /* get options */
    get_options(argc,argv,&rundata);

    printf("[%d]: (%d,%d) \n", rank, rundata.myx, rundata.myy);
    
    /*
     * allocate our local arrays
     */
    locdens = array2d(rundata.localnx, rundata.localny);
    locvel  = array3d(2, rundata.localnx, rundata.localny);
    printf("[%d]: Allocated arrays\n", rank);
 
    fillarray2d(locdens, &rundata);
    fillarray3d(locvel, &rundata);
    printf("[%d]: Filled arrays\n", rank);

    if (rundata.localnx*rundata.localny < 200) 
        printarray2d(locdens, rundata.localnx, rundata.localny);
    writehdf5file(rundata, locdens, locvel);
    printf("[%d]: Wrote file\n", rank);

    freearray2d(locdens);
    freearray3d(locvel);

    ierr = MPI_Finalize();
    return 0;
}  


int get_options(int argc, char **argv, rundata_t *rundata) {
                                        
    struct option long_options[] = {
        {"nx",       required_argument, 0, 'x'},
        {"ny",       required_argument, 0, 'y'},
        {"npx",      required_argument, 0, 'X'},
        {"npy",      required_argument, 0, 'Y'},
        {"filename", required_argument, 0, 'f'},
        {"help",     no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    char c;
    int option_index;
    int tempint;
    int defaultnpts = 100;
    FILE *tst;
    char *defaultfname=rundata->filename;


    while (1) { 
        c = getopt_long(argc, argv, "x:y:f:h", long_options, 
                        &option_index);
        if (c == (char)-1) break;

        switch (c) { 
            case 0: if (long_options[option_index].flag != 0)
                    break;

            case 'x': tempint = atoi(optarg);
                      if (tempint < 1 || tempint > 500) {
                          fprintf(stderr,
                                  "%s: Cannot use number of points %s;\n",
                                  argv[0], optarg);
                          fprintf(stderr,"  Using %d\n", defaultnpts);
                          rundata->globalnx = defaultnpts;
                      } else {
                          rundata->globalnx = tempint;
                      }
                      break;

            case 'y': tempint = atoi(optarg);
                      if (tempint < 1 || tempint > 500) {
                          fprintf(stderr,
                                  "%s: Cannot use number of points %s;\n",
                                  argv[0], optarg);
                          fprintf(stderr,"  Using %d\n", defaultnpts);
                          rundata->globalny = defaultnpts;
                      } else {
                          rundata->globalny = tempint;
                      }
                      break;


            case 'X': tempint = atoi(optarg);
                      if (tempint < 1 || tempint > rundata->nprocs) {
                          fprintf(stderr,
                                  "%s: Cannot use number of processors in x direction %s;\n",
                                  argv[0], optarg);
                          fprintf(stderr,"  Using %d\n", rundata->npx);
                      } else if (rundata->nprocs % tempint != 0) {
                          fprintf(stderr,
                                  "%s: Number of processors in x direction %s does not divide %d;\n",
                                  argv[0], optarg, rundata->nprocs);
                          fprintf(stderr,"  Using %d\n", rundata->npx);
                      } else {
                          rundata->npx = tempint;
                          rundata->npy = rundata->nprocs / tempint;
                      }
                      break;

            case 'Y': tempint = atoi(optarg);
                      if (tempint < 1 || tempint > rundata->nprocs) {
                          fprintf(stderr,
                                  "%s: Cannot use number of processors in y direction %s;\n",
                                  argv[0], optarg);
                          fprintf(stderr,"  Using %d\n", rundata->npy);
                      } else if (rundata->nprocs % tempint != 0) {
                          fprintf(stderr,
                                  "%s: Number of processors in y direction %s does not divide %d;\n",
                                  argv[0], optarg, rundata->nprocs);
                          fprintf(stderr,"  Using %d\n", rundata->npy);
                      } else {
                          rundata->npy = tempint;
                          rundata->npx = rundata->nprocs / tempint;
                      }
                      break;

            case 'f': strncpy(rundata->filename, optarg, MAXFILENAME-1);
                      if (!(tst=fopen(rundata->filename,"w"))) {
                          fprintf(stderr,
                                  "Cannot use filename %s;\n",
                                  rundata->filename);
                          fprintf(stderr, "  Using %s\n",defaultfname);
                          strcpy(rundata->filename, defaultfname);
                      } else
                        fclose(tst);
                      break; 

            case 'h':
                  puts("Options: ");
                  puts("    --nx=N         (-x N): Set the number of grid cells in x direction.");
                  puts("    --ny=N         (-y N): Set the number of grid cells in y direction.");
                  puts("    --npx=N        (-X N): Set the number of processors in the x direction.");
                  puts("    --npy=N        (-Y N): Set the number of processors in the y direction.");
                  puts("    --fileaname=S  (-f S): Set the output filename.");
                  puts("");
                  return +1;

            default: printf("Invalid option %s\n", optarg);
                 break;
        }
    }

    rundata -> myy = rundata->rank / (rundata->npx);
    rundata -> myx = rundata->rank % (rundata->npx);
 
    rundata->localnx = (rundata->globalnx / rundata->npx);
    rundata->localny = (rundata->globalny / rundata->npy);

    /* last row/column gets any extra / fewer points to make things work out: */
    if (rundata->myx == rundata->npx-1) 
        rundata->localnx = (rundata->globalnx - (rundata->npx-1)*(rundata->localnx));
    if (rundata->myy == rundata->npy-1) 
        rundata->localny = (rundata->globalny - (rundata->npy-1)*(rundata->localny));

    return 0;
}

double **array2d(int nx, int ny) {
    int i;
    double *data = (double *)malloc(nx*ny*sizeof(double));
    double **p = (double **)malloc(nx*sizeof(double *));

    if (data == NULL) return NULL;
    if (p == NULL) {
        free(data);
        return NULL;
    }

    for (i=0; i<nx; i++) {
        p[i] = &(data[ny*i]);
    }
    
    return p;
}

void freearray2d(double **p) {
    free(p[0]);
    free(p);
    return;
}

double ***array3d(int nd, int nx, int ny) {
    int i;
    double *data = (double *)malloc(nd*nx*ny*sizeof(double));
    double **datap = (double **)malloc(nd*nx*sizeof(double *));
    double ***p = (double ***)malloc(nd*sizeof(double **));

    if (data == NULL) return NULL;
    if (datap == NULL) {
        free(data);
        return NULL;
    }
    if (p == NULL) {
        free(data);
        free(datap);
        return NULL;
    }

    for (i=0; i<nd*nx; i++) {
        datap[i] = &(data[ny*i]);
    }

    for (i=0; i<nd; i++) {
        p[i] = &(datap[nx*i]);
    }
    
    return p;
}

void freearray3d(double ***p) {
    free(p[0][0]);
    free(p[0]);
    free(p);
    return;
}

void fillarray2d(double **d, rundata_t *r) {
    int i,j;
    double r2,r2x;
    int gnx = r->globalnx;
    int gny = r->globalny;
    int nx = r->localnx;
    int ny = r->localny;
    int npx = r->npx;
    int npy = r->npy;
    int myx = r->myx;
    int myy = r->myy;
    int startx,starty;
    double sigma=gnx/4.;

    startx = (gnx/npx)*myx;
    starty = (gny/npy)*myy;

    for (i=0;i<nx;i++) {
        r2x = ((i+startx)-gnx/2.)*((i+startx)-gnx/2.);
        for (j=0;j<ny;j++) {
            r2 = r2x + ((j+starty)-gny/2.)*((j+starty)-gny/2.);
            d[i][j] = 1. + 4.*exp(-r2/(2.*sigma*sigma));
        }
    } 

    return;
}

void fillarray3d(double ***v, rundata_t *r) {
    int i,j;
    int gnx = r->globalnx;
    int gny = r->globalny;
    int nx = r->localnx;
    int ny = r->localny;
    int npx = r->npx;
    int npy = r->npy;
    int myx = r->myx;
    int myy = r->myy;
    int startx,starty;
    double sigma=gnx/4.;
    double r2,r2x;

    startx = (gnx/npx)*myx;
    starty = (gny/npy)*myy;

    for (i=0;i<nx;i++) {
        r2x = ((i+startx)-gnx/2.)*((i+startx)-gnx/2.);
        for (j=0;j<ny;j++) {
            r2 = r2x + ((j+starty)-gny/2.)*((j+starty)-gny/2.);
            if (r2 < 1.e-6) {
                v[0][i][j] = 0.;
                v[1][i][j] = 0.;
            } else {
                v[0][i][j] = exp(-r2/(2.*sigma*sigma))*((j+starty)-gny/2.);
                v[1][i][j] = -exp(-r2/(2.*sigma*sigma))*((i+startx)-gnx/2.);
            } 
        }
    } 

    return;
}


void printarray2d(double **d, int nx, int ny) {
    int i,j;

    for (i=0;i<nx;i++) {
        for (j=0;j<ny;j++) {
            printf("%10.3g\t",d[i][j]);
        }
        puts("");
    } 

    return;
}

void nearsquare(int nprocs, int *npx, int *npy) {
    int sq = ceil(sqrt((double)nprocs));
    int n,m;

    if (sq*sq == nprocs) {
        *npx = *npy = sq;
    } else {
        for (n = sq+1; n>=1; n--) {
            if (nprocs % n == 0) {
                m = nprocs/n;
                *npx = n; *npy = m;
                if (m<n) {*npx = m; *npy = n;}
                break;
            } 
        }
    }
    return;
}
