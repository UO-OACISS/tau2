/*
 *  (C) 2001 by Argonne National Laboratory
 *      See COPYRIGHT in top-level directory.
 */

/*
 *  @author  Bill Gropp, Anthony Chan, Wyatt Spear
 */


#include "trace_impl.h"
#if defined( STDC_HEADERS ) || defined( HAVE_STDIO_H )
#include <stdio.h>
#endif
#if defined( STDC_HEADERS ) || defined( HAVE_STDLIB_H )
#include <stdlib.h>
#endif
#if defined( STDC_HEADERS ) || defined( HAVE_STRING_H )
#include <string.h>
#endif
#include "trace_API.h"
#include <stddef.h>
#include <TAU_tf.h>

#define TRACEINPUT_SUCCESS 0
#define TRACEINPUT_FAIL    -1
#define DRAW_TRUE    1
#define DRAW_FALSE   0

#define  MAX_LEGEND_LEN  128
#define  MAX_LABEL_LEN   512
#define  MAX_LINE_LEN    1024
#define  MAX_CATEGORIES  128

int debugPrint = 0;
/*/bool multiThreaded = false;*/
#define dprintf if (debugPrint) printf


/* The choice of the following numbers is arbitrary */
#define TAU_SAMPLE_CLASS_TOKEN   71
#define TAU_DEFAULT_COMMUNICATOR 42
/* any unique id */

/* Define limits of sample data (user defined events).  Not used (yet).*/
struct {
  unsigned long long umin;
  unsigned long long umax;
} taulongbounds = { 0, (unsigned long long)~(unsigned long long)0 };

/* Define limits of sample data (user defined events).  Not used (yet).*/
struct {
  double fmin;
  double fmax;
} taufloatbounds = {-1.0e+300, +1.0e+300};

/* These structures are used in user defined event data */

/* Global data */
/*/0 = no threads.  1 = threaded*/
int multiThreaded = 0;
int sampgroupid = 0;
int sampclassid = 0; 
/*/Contains the offset to use for threaded process IDs*/
int *offset = 0; 
/*/Defines the number of y coordinate maps to create*/
int ycordsets = 0;

/*/Category head container*/
typedef struct {
    TRACE_Category_head_t *hdr;
    char                  *legend;
    char                  *label;
    int                    num_methods;
    int                   *methods;
} DRAW_Category;



#define  MAX_COLNAMES    10
#define  MAX_NAME_LEN    128

/*/Y coordinate map container*/
typedef struct {
    int               num_rows;
    int               num_columns;
    char              title_name[ MAX_NAME_LEN ];
    char              column_names[ MAX_COLNAMES ][ MAX_NAME_LEN ];
    int              *elems;
    int               num_methods;
    int              *methods;
} DRAW_YCoordMap;


/*/Primitive drawable object container*/
typedef struct {
    double            starttime;
    double            endtime;
    int               type_idx;
    int               num_info;
    char             *info;
    int               num_tcoords;
    double           *tcoords;
    int               num_ycoords;
    int              *ycoords;
} DRAW_Primitive;

/*/Trace file container*/
typedef struct _trace_file {
    Ttf_FileHandleT   fd;
    char              line[ MAX_LINE_LEN ];
    int               max_types;
    int               num_types;
	int				  event_count;
	int				  arrow_count;
    DRAW_Category   **types;
	DRAW_YCoordMap   *ymap;
	DRAW_Primitive   *prime;
} DRAW_File;

/*/Link object used in linked list (for tracking arrow and state objects)*/
struct event_stack
{
	int sid;
	int tid;
	int nid;
	
	int dtid;
	int dnid;
	int size;
	
	double intime;
    struct event_stack *next;
	struct event_stack *last;
};

/*/Linked list for state objects*/
struct event_stack *top=NULL;

/*/Linked list for arrow objects*/
struct event_stack *arrowTop=NULL;

/*/The current color to apply to an event or state*/
int curcolor=0;

/*the largest index seen so far*/
int maxidx = 0;

/*/Contains one element for each node, with each element holding the number of threads at that node*/
int *countthreads = NULL;

/* FIX GlobalID so it takes into account numthreads */
/* utilities*/
/*/int GlobalID(int localnodeid, int localthreadid);*/
int GlobalID(int localnodeid, int localthreadid)
{
  if (multiThreaded)
  {
    if (offset == (int *) NULL)
    {
      printf("Error: offset vector is NULL in GlobalId()\n");
      return localnodeid;
    }
    
    /*/ for multithreaded programs, modify this routine */
    return offset[localnodeid]+localthreadid;  /*/ for single node program */
  }
  else
  { 
    return localnodeid;
  }
}  


/*/Swaps bytes for bigendian systems*/
void bswp_byteswap( const int    Nelem,
                    const int    elem_sz,
                          char  *bytes )
{
    char *bptr;
    char  btmp;
    int end_ii;
    int ii, jj;

    bptr = bytes;
    for ( jj = 0; jj < Nelem; jj++ ) {
         for ( ii = 0; ii < elem_sz/2; ii++ ) {
             end_ii          = elem_sz - 1 - ii;
             btmp            = bptr[ ii ];
             bptr[ ii ]      = bptr[ end_ii ];
             bptr[ end_ii ] = btmp;
         }
         bptr += elem_sz;
    }
}

/*/Allocates a YCoordMap object*/
DRAW_YCoordMap *YCoordMap_alloc( int Nrows, int Ncols, int Nmethods )
{
    DRAW_YCoordMap   *map;

    map               = (DRAW_YCoordMap *) malloc( sizeof(DRAW_YCoordMap) );

    map->num_rows     = Nrows;
    map->num_columns  = Ncols;
    if ( Nrows * Ncols > 0 )
        map->elems    = (int *) malloc( Nrows * Ncols * sizeof( int ) );
    else
        map->elems    = NULL;

    map->num_methods  = Nmethods;
    if ( Nmethods > 0 )
        map->methods  = (int *) malloc( Nmethods * sizeof( int ) );
    else
        map->methods  = NULL;
    return map;
}
/*/Destroys a YCoordMap object*/
void YCoordMap_free( DRAW_YCoordMap *map )
{
    if ( map != NULL ) {
	       if ( map->methods != NULL ) {
            free( map->methods );
            map->methods = NULL;
        }
        if ( map->elems != NULL ) {
            free( map->elems );
            map->elems = NULL;
        }
        free( map );
    }
}


/* legend_len & label_len are lengths of the string withOUT counting NULL */
/*/Allocates a Category object*/
DRAW_Category *Category_alloc( int legend_len, int label_len, int Nmethods )
{
    DRAW_Category    *type;

    type              = (DRAW_Category *) malloc( sizeof(DRAW_Category) );
    type->hdr         = (TRACE_Category_head_t *)
                        malloc( sizeof(TRACE_Category_head_t) );

    if ( legend_len > 0 )
        type->legend  = (char *) malloc( (legend_len+1) * sizeof(char) );
    else
        type->legend  = NULL;

    if ( label_len > 0 )
        type->label   = (char *) malloc( (label_len+1) * sizeof(char) );
    else
        type->label   = NULL;

    type->num_methods  = Nmethods;
    if ( Nmethods > 0 )
        type->methods = (int *) malloc( Nmethods * sizeof( int ) );
    else
        type->methods = NULL;
    return type;
}

/*/Destroys a Category Object*/
void Category_free( DRAW_Category *type )
{
    if ( type != NULL ) {
        if ( type->methods != NULL ) {
            free( type->methods );
            type->methods = NULL;
        }
        if ( type->label != NULL ) {
            free( type->label );
            type->label = NULL;
        }
        if ( type->legend != NULL ) {
            free( type->legend );
            type->legend = NULL;
        }
        if ( type->hdr != NULL ) {
            free( type->hdr );
            type->hdr = NULL;
        }
        free( type );
    }
}

/*/Copies a category head object to a new one.*/
void Category_head_copy(       TRACE_Category_head_t *hdr_copy,
                         const TRACE_Category_head_t *hdr_copier )
{
    if ( hdr_copy != NULL && hdr_copier != NULL ) {
        hdr_copy->index  = hdr_copier->index;
        hdr_copy->shape  = hdr_copier->shape;
        hdr_copy->red    = hdr_copier->red  ;
        hdr_copy->green  = hdr_copier->green;
        hdr_copy->blue   = hdr_copier->blue ;
        hdr_copy->alpha  = hdr_copier->alpha;
        hdr_copy->width  = hdr_copier->width;
    }
}

/*/Allocates a new primitive object*/
DRAW_Primitive *Primitive_alloc( int num_vtxs )
{
    DRAW_Primitive    *prime;

    prime               = (DRAW_Primitive *) malloc( sizeof(DRAW_Primitive) );
    prime->num_info     = 0;
    prime->info         = NULL;
    prime->num_tcoords  = num_vtxs;
    prime->tcoords      = (double *) malloc( num_vtxs * sizeof(double) );
    prime->num_ycoords  = num_vtxs;
    prime->ycoords      = (int *) malloc( num_vtxs * sizeof(int) );
    return prime;
}

/*/Destroys a primitive object*/
void Primitive_free( DRAW_Primitive *prime )
{
    if ( prime != NULL ) {
        if ( prime->num_info > 0 && prime->info != NULL ) {
            free( prime->info );
            prime->num_info  = 0;
            prime->info      = NULL;
        }
        if ( prime->num_tcoords > 0 && prime->tcoords != NULL ) {
            free( prime->tcoords );
            prime->num_tcoords  = 0;
            prime->tcoords      = NULL;
        }
        if ( prime->num_ycoords > 0 && prime->ycoords != NULL ) {
            free( prime->ycoords );
            prime->num_ycoords  = 0;
            prime->ycoords      = NULL;
        }
        free( prime );
    }
}


/*/Determines if valid data has been read by the callback routines*/
int thispeak = 0;
/*/The number of End _ of _ File messages in the TAU trace*/
int numstops = 0;
/*/Used to keep track of the number of end-messages seen so far*/
int countstops  = 0;
/*/The ID of the current kind of event*/
int currentkind = -1;
/*/The total number of nodes*/
int maxnode = 0;
/*/The largest number of threads at any one node*/
int maxthrd = 0;
/*/The total number of state types (includes events and arrows)*/
int tot_types = 0;
/*/The number of threads */
int numthreads = 0;
/*/0 = trace still in progress, 1 = the trace is really over*/
int EndOfTrace = 0;
/*/The clock period of the TAU trace*/
double clockP = 1;
/*/The callback holder*/
Ttf_CallbacksT cb;


/*TAU Callback Routines*/
/***************************************************************************
 * Description: ClockPeriod (in microseconds) is specified here. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int ClockPeriod( void*  userData, double clkPeriod )
{
  dprintf("Clock period %g\n", clkPeriod);
  clockP = clkPeriod;

  return 0;
}
int rows = 0;
/***************************************************************************
 * Description: DefThread is called when a new nodeid/threadid is encountered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefThread(void *userData, unsigned int nodeToken, unsigned int threadToken,
const char *threadName )
{
	int i;
 dprintf("DefThread nid %d tid %d, thread name %s\n", 
		  nodeToken, threadToken, threadName);
    numthreads++;/*/Increment the total number of threads*/
  /*/If this is the first thread entry we see*/
  if(countthreads == NULL)
  {/*/Create a new int array for count threads*/
	countthreads = (int *) malloc((nodeToken+1)*sizeof(int));
	/*/Set the thread count for this node*/
	countthreads[nodeToken] = threadToken;

	for(i = 0; i<nodeToken;i++)
		countthreads[i] = 0;
  }
  else
  {/*/Otherwise expand the size of the array and increase the thread counts as necessary*/
	if(nodeToken<maxnode)
	{
		if(countthreads[nodeToken] < threadToken)
			countthreads[nodeToken] = threadToken;
	}
	else
	{
		int i;
		int *temp = (int *) malloc((nodeToken+1)*sizeof(int));
		for(i  = 0; i<nodeToken; i++)
		{
			temp[i] = countthreads[i];
		}
		temp[nodeToken] = threadToken;
		free(countthreads);
		countthreads = NULL;
		countthreads = temp;
		temp = NULL;
		}
  }
  /*/Set the maximum and minimum node size observed (Unused?!)*/
  if(nodeToken > maxnode)
	maxnode = nodeToken;
	
	if(threadToken > maxthrd)
		maxthrd = threadToken;
		/*/If we have any threads this is a threaded program*/
    

  if (threadToken > 0) multiThreaded = 1; 
  
  rows++;/*/Redundant with numthreaeds?!*/

  
  return 0;
}


/***************************************************************************
 * Description: EndTrace is called when an EOF is encountered in a tracefile.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EndTrace( void *userData, unsigned int nodeToken, unsigned int threadToken)
{
  dprintf("EndTrace nid %d tid %d\n", nodeToken, threadToken);
  countstops += 1; /*/Count the total number of eof markers (first pass)*/
  numstops -= 1;  /*//Count down through all eof markers (second pass)*/
  if(numstops <= 0)
  {
	currentkind = (int)TRACE_EOF; 
	thispeak = 1; 
  }
  
  if(numthreads <= countstops)
	EndOfTrace = 1;

  return 0;
}

int *eventtypes = NULL;
int eventcount = 0;

/***************************************************************************
 * Description: DefUserEvent is called to register the name and a token of the
 *  		user defined event (or a sample event in Vampir terminology).
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefUserEvent( void *userData, unsigned int userEventToken,
		const char *userEventName , int monotonicallyIncreasing)
{
int dodifferentiation;
  char *name = strdup(userEventName);
  int len = strlen(name);
  	    int             type_idx = userEventToken;
    int             legend_len;
    char            label[MAX_LABEL_LEN];
    int             label_len;
    char            str4methods[MAX_LABEL_LEN];
    int             methods_len;
	DRAW_Category  *type;
    TRACE_Category_head_t *hdr;
    char *legend;
  dprintf("DefUserEvent event id %d user event name %s\n", userEventToken,
		  userEventName);

  /* We need to remove the backslash and quotes from "\"funcname\"" */

  if ((name[0] == '"' ) && (name[len-1] == '"'))
  {
     name += 1;
     name[len-2] = '\0';
  }
dprintf("TestUDef1\n");
  /* create a state record */
  if (!monotonicallyIncreasing)
  {

    dodifferentiation = 1; 
  }
  /* non monotonically increasing record ids are kept in a list for identification later*/
	else 
	{
		int *temp;
		int i;
		dodifferentiation = 0; 
		eventcount++;
		temp = (int *)malloc(eventcount*sizeof(int));
		
		for(i = 0; i<eventcount-1; i++)
		{
			temp[i] = eventtypes[i];
		}
		temp[eventcount-1] = userEventToken;
		free(eventtypes);
		eventtypes = temp;
		temp = NULL;
	}
	

    /*/int             line_pos;*/
    /*/int             lgd_pos, lbl_pos;*/
   /*/ char           *newline;*/
    /*/char           *info_A;  *info_B;*/
    legend = malloc((strlen(name)+1));

      memcpy(legend,  name, strlen(name)+1);
    legend_len  = strlen( legend );
    /*/newline     = (char *) (((TRACE_file)userData)->line + line_pos);*/

        strncpy( label, "Count=%d", MAX_LABEL_LEN );
		/*/sprintf( label, "%c", '\0' );*/
		/*/dprintf( "Yo! <%s>\n", label);*/

        label_len = strlen( label );
        methods_len = 0;

    type = Category_alloc( legend_len, label_len, methods_len );
    hdr  = type->hdr;

    /* Set the Output Parameters of the routine */
    hdr->index = type_idx;

       hdr->shape = TRACE_SHAPE_EVENT;
   
   curcolor++;
   if(curcolor > 21)
	curcolor = 1;
   /*/dprintf("color %d\n",EndOfTrace);*/
   switch(curcolor)
   {
   case 1:
   
		hdr->red    = 0;/*/red;*/
		hdr->green  = 0;/*/green;*/
		hdr->blue   = 255;/*/blue;*/
		break;
	
	case 2:
   
		hdr->red    = 0;
		hdr->green  = 255;
		hdr->blue   = 0;
		break;
	
	case 3:
   
		hdr->red    = 255;
		hdr->green  = 0;
		hdr->blue   = 0;
		break;
	
	case 4:
   
		hdr->red    = 0;
		hdr->green  = 255;
		hdr->blue   = 255;
		break;
	
	case 5:
   
		hdr->red    = 255;
		hdr->green  = 0;
		hdr->blue   = 255;
		break;
	
	case 6:
   
		hdr->red    = 255;
		hdr->green  = 255;
		hdr->blue   = 0;
		break;
	
	case 7:
   
		hdr->red    = 128;
		hdr->green  = 255;
		hdr->blue   = 128;
		break;
	
	case 8:
   
		hdr->red    = 128;
		hdr->green  = 128;
		hdr->blue   = 255;
		break;
	
	case 9:
   
		hdr->red    = 255;
		hdr->green  = 128;
		hdr->blue   = 128;
		break;
	
	case 10:
   
		hdr->red    = 128;
		hdr->green  = 255;
		hdr->blue   = 0;
		break;
	
		case 11:
   
		hdr->red    = 0;
		hdr->green  = 128;
		hdr->blue   = 255;
		break;
	
		case 12:
   
		hdr->red    = 255;
		hdr->green  = 0;
		hdr->blue   = 128;
		break;
		
		case 13:
   
		hdr->red    = 128;
		hdr->green  = 0;
		hdr->blue   = 255;
		break;
		
		case 14:
   
		hdr->red    = 0;
		hdr->green  = 255;
		hdr->blue   = 128;
		break;
		
		case 15:
   
		hdr->red    = 255;
		hdr->green  = 128;
		hdr->blue   = 0;
		break;
		
		case 16:
   
		hdr->red    = 0;
		hdr->green  = 128;
		hdr->blue   = 0;
		break;

		case 17:
   
		hdr->red    = 0;
		hdr->green  = 0;
		hdr->blue   = 128;
		break;
		
		case 18:
		
		hdr->red    = 128;
		hdr->green  = 0;
		hdr->blue   = 0;
		break;
		
		case 19:
   
		hdr->red    = 128;
		hdr->green  = 0;
		hdr->blue   = 128;
		break;

		case 20:
   
		hdr->red    = 128;
		hdr->green  = 128;
		hdr->blue   = 0;
		break;
		
		case 21:
		
		hdr->red    = 0;
		hdr->green  = 128;
		hdr->blue   = 128;
		break;
	
	default:
	{   printf("Bad %d\n", curcolor);
		hdr->red    = 255;
		hdr->green  = 255;
		hdr->blue   = 255;
	}
	}
	
    hdr->alpha  = 255;
    hdr->width  = 20;

    if ( legend_len > 0 )
        strcpy( type->legend, legend );
    if ( label_len > 0 )
        strcpy( type->label, label );
    if ( methods_len > 0 )
        /* Assume 1 method ID */
        type->methods[ 0 ] = atoi( str4methods );

    /*if ( ((TRACE_file)userData)->num_types >= ((TRACE_file)userData)->max_types )
        return 10;*/
    if ( ((TRACE_file)userData)->num_types >= ((TRACE_file)userData)->max_types )
  	{
		((TRACE_file)userData)->max_types=(((TRACE_file)userData)->max_types)*2;
		((TRACE_file)userData)->types      = (DRAW_Category **) realloc(((TRACE_file)userData)->types, ((TRACE_file)userData)->max_types
                                              * sizeof(DRAW_Category *) );
		  
 	 } 
    ((TRACE_file)userData)->types[ ((TRACE_file)userData)->num_types ] = type;

	((TRACE_file)userData)->num_types++;

  currentkind = (int)TRACE_CATEGORY;
  thispeak = 1;
 /* for non monotonically increasing data (not yet implemented) */

  if(userEventToken > maxidx)
	maxidx = userEventToken;

  return 0;
}

int trigcount =  0;
/***************************************************************************
 * Description: EventTrigger is called when a user defined event is triggered.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EventTrigger( void *userData, double time, 
		unsigned int nodeToken,
		unsigned int threadToken,
	       	unsigned int userEventToken,
		long long userEventValue)
{

	DRAW_Category  *type;
    DRAW_Primitive *prime;
    int             type_idx = userEventToken;

    int             num_vertices;
    int            infovals[1];
    int             idx;		
int i;
for(i = 0; i<eventcount;i++)
{
	if(eventtypes[i] == userEventToken)
	{
		i = -1;
		break;
	}
}

if(i != -1)
{
	if(trigcount == 0)
	{
		trigcount++;
		return 0;
	}
	if(trigcount == 2)
	{
		trigcount = 0;
		return 0;
	}
	trigcount = 2;
}
  dprintf(
  "EventTrigger: time %g, nid %d tid %d event id %d triggered value %lld \n", 
  time, nodeToken, threadToken, userEventToken, userEventValue);

   
		   
		   
#if defined( DEBUG )
    printf( "%s %lf %lf %d ", typename, starttime, endtime, type_idx );
#endif
   

    num_vertices = 1; 
    /* Search for the valid Category in the category table */
    type = NULL;
    for ( idx = 0; idx < tot_types; idx++ ) {
        if ( ((TRACE_file)userData)->types[ idx ]->hdr->index == type_idx ) {
            type = ((TRACE_file)userData)->types[ idx ];
            break;
        }
    }

infovals[0] = (int)userEventValue;

    /* Allocate a new Primitive */
    prime = Primitive_alloc( num_vertices );
    prime->starttime  = time*clockP;
    prime->endtime    = time*clockP;
    prime->type_idx  = type_idx;
    
	
	
        prime->num_info  = 4;
        prime->info      = (char *) malloc( prime->num_info * sizeof(char) );
        memcpy( prime->info, infovals, prime->num_info );
#if ! defined( WORDS_BIGENDIAN )
		
        bswp_byteswap( 1, sizeof( int ), prime->info );
		
#endif

    prime->num_tcoords = num_vertices;
    prime->num_ycoords = num_vertices;
	
	
	
	prime->ycoords[ 0 ] = GlobalID(nodeToken,threadToken);
	prime->tcoords[ 0 ] = time*clockP;
	
	/* Free the previous allocated Primitive stored at TRACE_file */
    Primitive_free( ((TRACE_file)userData)->prime );
    ((TRACE_file)userData)->prime = prime;





  /* write the sample data */
  currentkind = (int)TRACE_PRIMITIVE_DRAWABLE;
  thispeak = 1;
  return 0;
}

/***************************************************************************
 * Description: DefStateGroup registers a profile group name with its id.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefStateGroup( void *userData, unsigned int stateGroupToken, 
		const char *stateGroupName )
{
  dprintf("StateGroup groupid %d, group name %s\n", stateGroupToken, 
		  stateGroupName);

  /* create a default activity (group) */
 
  return 0;
}

/***************************************************************************
 * Description: DefState is called to define a new symbol (event). It uses
 *		the token used to define the group identifier. 
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int DefState( void *userData, unsigned int stateToken, const char *stateName, 
		unsigned int stateGroupToken )
{
  

  /* We need to remove the backslash and quotes from "\"funcname\"" */
  char *name = strdup(stateName);
  int len = strlen(name);
  DRAW_Category  *type;
    TRACE_Category_head_t *hdr;

    int             type_idx = stateToken;
    int             line_pos;

    char           *newline;
    
  int             legend_len;
    char            label[MAX_LABEL_LEN];
    int             label_len;
    char            str4methods[MAX_LABEL_LEN];
    int             methods_len;
    char            *legend;
  
  dprintf("DefState stateid %d stateName %s stategroup id %d\n",
		  stateToken, stateName, stateGroupToken);
  
  if ((name[0] == '"' ) && (name[len-1] == '"'))
  {
     name += 1;
     name[len-2] = '\0';
  }

  /* create a state record */
 /* if(cathold = NULL)
  {
	cathold = (DRAW_CatStack)malloc(sizeof(struct cat_stack));
  }*/
  
  
	

    legend = malloc((strlen(name)+1));
    

	memcpy(legend,  name, strlen(name)+1);
    legend_len  = strlen( legend );
    newline     = (char *) (((TRACE_file)userData)->line + line_pos);

        label_len = 0;
#if defined( DEBUG )
    printf( "\n" );
    fflush( NULL );
#endif

        methods_len = 0;
#if defined( DEBUG )
    printf( "\n" );
    fflush( NULL );
#endif

    type = Category_alloc( legend_len, label_len, methods_len );
    hdr  = type->hdr;

    /* Set the Output Parameters of the routine */
    hdr->index = type_idx;

        hdr->shape = TRACE_SHAPE_STATE;
     
   curcolor++;
   if(curcolor > 21)
	curcolor = 1;
   dprintf("color %d\n",curcolor);
   switch(curcolor)
   {
   case 1:
   
		hdr->red    = 0;/*/red;*/
		hdr->green  = 0;/*/green;*/
		hdr->blue   = 255;/*/blue;*/
		break;
	
	case 2:
   
		hdr->red    = 0;
		hdr->green  = 255;
		hdr->blue   = 0;
		break;
	
	case 3:
   
		hdr->red    = 255;
		hdr->green  = 0;
		hdr->blue   = 0;
		break;
	
	case 4:
   
		hdr->red    = 0;
		hdr->green  = 255;
		hdr->blue   = 255;
		break;
	
	case 5:
   
		hdr->red    = 255;
		hdr->green  = 0;
		hdr->blue   = 255;
		break;
	
	case 6:
   
		hdr->red    = 255;
		hdr->green  = 255;
		hdr->blue   = 0;
		break;
	
	case 7:
   
		hdr->red    = 128;
		hdr->green  = 255;
		hdr->blue   = 128;
		break;
	
	case 8:
   
		hdr->red    = 128;
		hdr->green  = 128;
		hdr->blue   = 255;
		break;
	
	case 9:
   
		hdr->red    = 255;
		hdr->green  = 128;
		hdr->blue   = 128;
		break;
	
	case 10:
   
		hdr->red    = 128;
		hdr->green  = 255;
		hdr->blue   = 0;
		break;
	
		case 11:
   
		hdr->red    = 0;
		hdr->green  = 128;
		hdr->blue   = 255;
		break;
	
		case 12:
   
		hdr->red    = 255;
		hdr->green  = 0;
		hdr->blue   = 128;
		break;
		
		case 13:
   
		hdr->red    = 128;
		hdr->green  = 0;
		hdr->blue   = 255;
		break;
		
		case 14:
   
		hdr->red    = 0;
		hdr->green  = 255;
		hdr->blue   = 128;
		break;
		
		case 15:
   
		hdr->red    = 255;
		hdr->green  = 128;
		hdr->blue   = 0;
		break;
		
		case 16:
   
		hdr->red    = 0;
		hdr->green  = 128;
		hdr->blue   = 0;
		break;

		case 17:
   
		hdr->red    = 0;
		hdr->green  = 0;
		hdr->blue   = 128;
		break;
		
		case 18:
		
		hdr->red    = 128;
		hdr->green  = 0;
		hdr->blue   = 0;
		break;
		
		case 19:
   
		hdr->red    = 128;
		hdr->green  = 0;
		hdr->blue   = 128;
		break;

		case 20:
   
		hdr->red    = 128;
		hdr->green  = 128;
		hdr->blue   = 0;
		break;
		
		case 21:
		
		hdr->red    = 0;
		hdr->green  = 128;
		hdr->blue   = 128;
		break;
	
	default:
	{   printf("Bad %d\n", curcolor);
		hdr->red    = 255;
		hdr->green  = 255;
		hdr->blue   = 255;
	}
	}
	
	
	
	
	
    hdr->alpha  = 255;
    hdr->width  = 1;

    if ( legend_len > 0 )
        strcpy( type->legend, legend );
		
    if ( label_len > 0 )
        strcpy( type->label, label );
    if ( methods_len > 0 )
        /* Assume 1 method ID */
        type->methods[ 0 ] = atoi( str4methods );

    if ( ((TRACE_file)userData)->num_types >= ((TRACE_file)userData)->max_types )
  	{
		((TRACE_file)userData)->max_types=(((TRACE_file)userData)->max_types)*2;
		((TRACE_file)userData)->types      = (DRAW_Category **) realloc(((TRACE_file)userData)->types, ((TRACE_file)userData)->max_types
                                              * sizeof(DRAW_Category *) );
		  
 	 }      
/*		return 10;*/
		
    ((TRACE_file)userData)->types[ ((TRACE_file)userData)->num_types ] = type;



((TRACE_file)userData)->num_types++;



  currentkind = (int)TRACE_CATEGORY;
  thispeak = 1;
 
  if(maxidx < stateToken)
    maxidx = stateToken;

  return 0;
}


/***************************************************************************
 * Description: EnterState is called at routine entry by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int EnterState(void *userData, double time, 
		unsigned int nodeid, unsigned int tid, unsigned int stateid)
{
  
		 
	struct event_stack *temp;
    temp=
  (struct event_stack *)malloc(sizeof(struct event_stack));
    temp->sid=stateid;
	temp->tid = tid;
	temp->nid = nodeid;
	temp->intime = time*clockP;
    temp->next=top;
	temp->last=NULL;
	
	dprintf("Entered state %d time %g nid %d tid %d\n", 
		  stateid, time, nodeid, tid);
	
	if(top != NULL)
	{
		top->last=temp;
	}
	
	top=temp; 
	((TRACE_file)userData)->event_count++;
		  
  thispeak = 0;
  return 0;
}

/***************************************************************************
 * Description: LeaveState is called at routine exit by trace input library
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int LeaveState(void *userData, double time, unsigned int nid, unsigned int tid)
{
  
  
  
	struct event_stack *temp = top;
	int found = 0;
	DRAW_Category  *type;
    DRAW_Primitive *prime;
	int             type_idx;
    int             num_vertices;
    int             idx;
	
	dprintf("Leaving state time %g nid %d tid %d\n", time, nid, tid);
dprintf("%d events!\n", ((TRACE_file)userData)->event_count);
	
	while(found == 0)
	{
		if(temp == NULL){dprintf("BROKE!\n"); break;}
		
		if((nid == temp->nid) && (tid == temp->tid))
		{
			found = 1;
			
			if(temp->next != NULL)
				temp->next->last = temp->last;
			
			if(temp->last != NULL && temp->next != NULL)
				temp->last->next = temp->next; 
				
			if(temp->last == NULL)
				top = temp->next;
				
			
			((TRACE_file)userData)->event_count--;
		}
		else
		{
			temp = temp->next;
		}
		
	}

	type_idx = temp->sid;

#if defined( DEBUG )
    printf( "%s %lf %lf %d ", typename, starttime, endtime, type_idx );
#endif
   
    num_vertices = 2; 

	
    /* Search for the valid Category in the category table */
    type = NULL;
    for ( idx = 0; idx < tot_types; idx++ ) {
        if ( ((TRACE_file)userData)->types[ idx ]->hdr->index == type_idx ) {
            type = ((TRACE_file)userData)->types[ idx ];
            break;
        }
    }

    /* Allocate a new Primitive */
    prime = Primitive_alloc( num_vertices );
    prime->starttime  = temp->intime;
    prime->endtime    = time*clockP;
    prime->type_idx  = type_idx;
    
	/*if ( *num_bytes > 0 ) {
        prime->num_info  = *num_bytes;
        prime->info      = (char *) malloc( prime->num_info * sizeof(char) );
        memcpy( prime->info, infovals, prime->num_info );
#if ! defined( WORDS_BIGENDIAN )
        bswp_byteswap( 2, sizeof( int ), prime->info );
#endif
    }*/
    prime->num_tcoords = num_vertices;
    prime->num_ycoords = num_vertices;
	prime->ycoords[ 0 ] = GlobalID(nid,tid);
	prime->tcoords[ 0 ] = (double)temp->intime;
	prime->ycoords[ 1 ] = GlobalID(nid,tid);
	prime->tcoords[ 1 ] = time*clockP;
    
    /* Free the previous allocated Primitive stored at TRACE_file */
    Primitive_free( ((TRACE_file)userData)->prime );
    ((TRACE_file)userData)->prime = prime;


	free(temp);
  
  currentkind = (int)TRACE_PRIMITIVE_DRAWABLE;
  thispeak = 1;
  return 0;
}



/***************************************************************************
 * Description: RegMessage is called when a message is sent by a process.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int RegMessage( void *userData, double time, 
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag )
{
	int found = 0;
char *name = "message";
	DRAW_Category  *type;
    TRACE_Category_head_t *hdr;
    int             legend_len;
    char            label[MAX_LABEL_LEN];
    int             label_len;
    char            str4methods[MAX_LABEL_LEN];
    int             methods_len;
    int             type_idx;
    char *legend;
	int i;
    for(i = 0; i < ((TRACE_file)userData)->num_types; i++ ) 
	{
        if ( ((TRACE_file)userData)->types[ i ]->hdr->index == messageTag+maxidx ) 
		{
			found = 1;
            break;
        }
	}

	if(found == 1 && ((TRACE_file)userData)->num_types != 0)
	{
		return 0;
	}
		
	
    type_idx = messageTag+maxidx;
    legend=malloc(strlen(name)+1);


    	memcpy(legend,  name, strlen(name)+1);
    legend_len  = strlen( legend );

    /* Set InfoKeys */        
	strncpy( label, "msg_tag=%d, msg_size=%d", MAX_LABEL_LEN );


	label_len = strlen( label );
	
	methods_len = 0;

    type = Category_alloc( legend_len, label_len, methods_len );
    hdr  = type->hdr;

    /* Set the Output Parameters of the routine */
    hdr->index = type_idx;

	hdr->shape = TRACE_SHAPE_ARROW;
	hdr->red    = 255;
	hdr->green  = 255;
	hdr->blue   = 255;
    hdr->alpha  = 255;
    hdr->width  = 3;

	
    if ( legend_len > 0 )
        strcpy( type->legend, legend );
    if ( label_len > 0 )
	{
        strcpy( type->label, label );
		dprintf("ArrowReg: %s\n", type->label);
	}
    if ( methods_len > 0 )
        /* Assume 1 method ID */
        type->methods[ 0 ] = atoi( str4methods );

   /* if ( ((TRACE_file)userData)->num_types >= ((TRACE_file)userData)->max_types )
        return 10;*/
	if ( ((TRACE_file)userData)->num_types >= ((TRACE_file)userData)->max_types )
  	{
		((TRACE_file)userData)->max_types=(((TRACE_file)userData)->max_types)*2;
		((TRACE_file)userData)->types      = (DRAW_Category **) realloc(((TRACE_file)userData)->types, ((TRACE_file)userData)->max_types
                                              * sizeof(DRAW_Category *) );
		  
 	 } 
    ((TRACE_file)userData)->types[ ((TRACE_file)userData)->num_types ] = type;

	((TRACE_file)userData)->num_types++;
	currentkind = (int)TRACE_CATEGORY;
	return 0;

}
struct event_stack *arrowBase = NULL;
/***************************************************************************
 * Description: SendMessage is called when a message is sent by a process.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int SendMessage( void *userData, double time, 
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag )
{
 
		 
	struct event_stack *temp;
	
    temp=
  (struct event_stack *)malloc(sizeof(struct event_stack));
    temp->sid=messageTag+maxidx;
	temp->tid = sourceThreadToken;
	temp->nid = sourceNodeToken;
	
	temp->dnid = destinationNodeToken;
	temp->dtid = destinationThreadToken;
	temp->size = messageSize;
	
	temp->intime = time*clockP;
    temp->next=arrowTop;
	temp->last=NULL;
	
	
	 dprintf("SendMessage: time %g, source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", 
		  time, 
		  sourceNodeToken, sourceThreadToken,
		  destinationNodeToken, destinationThreadToken,
		  messageSize, messageTag);
	
	if(arrowTop != NULL)
	{
		arrowTop->last=temp;
	}
	
	if(arrowBase == NULL)
	  arrowBase = temp;

	arrowTop=temp; 
	((TRACE_file)userData)->arrow_count++;
  thispeak = 0;
  return 0;
}

/***************************************************************************
 * Description: RecvMessage is called when a message is received by a process.
 * 		This is a callback routine which must be registered by the 
 * 		trace converter. 
 ***************************************************************************/
int RecvMessage( void *userData, double time,
		unsigned int sourceNodeToken,
		unsigned int sourceThreadToken, 
		unsigned int destinationNodeToken,
		unsigned int destinationThreadToken,
		unsigned int messageSize,
		unsigned int messageTag )
{
	
	struct event_stack *temp = arrowBase;
	int found = 0;
		DRAW_Category  *type;
    DRAW_Primitive *prime;

    int             type_idx;
    int             num_vertices;
    int             infovals[2];
    int             idx;
	
  dprintf("RecvMessage: time %g,source nid %d tid %d, destination nid %d tid %d, size %d, tag %d\n", 
		  time, 
		  sourceNodeToken, sourceThreadToken,
		  destinationNodeToken, destinationThreadToken,
		  messageSize, messageTag);

dprintf("%d arrows!\n", ((TRACE_file)userData)->arrow_count);

  
	
	while(found == 0)
	{
		if(temp == NULL){printf("BROKEN ARROW!\n"); break;}
		
		if((sourceNodeToken == temp->nid) && (sourceThreadToken == temp->tid) &&
		(destinationNodeToken == temp->dnid) && (destinationThreadToken == temp->dtid) &&
		(messageTag+maxidx == temp->sid) && (messageSize == temp->size))
		{
			found = 1;
			
			if(temp->next != NULL)
				temp->next->last = temp->last;
			
			if(temp->last != NULL && temp->next != NULL)
				temp->last->next = temp->next; 
				
			if(temp->last == NULL)
				arrowTop = temp->next;
			if(temp->next == NULL)
			  {
			    if(temp->last != NULL)
			      temp->last->next = NULL;

			    arrowBase = temp->last;
			  }
			
			((TRACE_file)userData)->arrow_count--;
		}
		else
		{
			temp = temp->last;
		}
		
	}
	
	
	type_idx = temp->sid;



    num_vertices = 2; 

    /* Search for the valid Category in the category table */
    type = NULL;
    for ( idx = 0; idx < tot_types; idx++ ) {
        if ( ((TRACE_file)userData)->types[ idx ]->hdr->index == type_idx ) {
            type = ((TRACE_file)userData)->types[ idx ];
            break;
        }
    }


	infovals[0] = messageTag;
	infovals[1] = messageSize;
	
    /* Allocate a new Primitive */
    prime = Primitive_alloc( num_vertices );
    prime->starttime  = temp->intime;
    prime->endtime    = time*clockP;
    prime->type_idx  = type_idx;
    
	

        prime->num_info  = 8;
        prime->info      = (char *) malloc(prime->num_info * sizeof(char) );
        memcpy( prime->info, infovals, prime->num_info);

#if ! defined( WORDS_BIGENDIAN )
        bswp_byteswap( 2, sizeof( int ), prime->info );
#endif
    prime->num_tcoords = num_vertices;
    prime->num_ycoords = num_vertices;
	prime->ycoords[ 0 ] = GlobalID(temp->nid,temp->tid);
	prime->tcoords[ 0 ] = temp->intime;
	prime->ycoords[ 1 ] = GlobalID(temp->dnid,temp->dtid);
	prime->tcoords[ 1 ] = time*clockP;
	Primitive_free( ((TRACE_file)userData)->prime );
    ((TRACE_file)userData)->prime = prime;

    

	free(temp);
  
  currentkind = (int)TRACE_PRIMITIVE_DRAWABLE;
  dprintf("Current: %d\n",currentkind);

thispeak = 1;
  return 0;
}


/****************************************************************************@
* TRACE_Open - Open a trace file for input
*
*  Input Parameter:
*. filespec - Name of file (or files; see below) to open. 
*
*  Output Parameter:
*. fp - Trace file handle (see Notes).
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
****************************************************************************@*/
TRACE_EXPORT
int TRACE_Open( const char filespec[], TRACE_file *fp )
{
	  Ttf_CallbacksT firstpass;
	TRACE_file tr;
	char *trc;
	char *edf;
	char *contain;
	  int recs_read = 0; 
	 if ( strncmp( filespec, "-h", 2 ) == 0 ) 
	 {
        *fp  = NULL;
        return 0;
     }
     
	contain=malloc(strlen(filespec)+1);
	tr = (TRACE_file) malloc( sizeof(struct _trace_file) );
	
	
	memcpy(contain, filespec, strlen(filespec)+1);
	trc = strtok(contain, ":");
	edf = strtok(NULL, ":");
	dprintf("Opening: %s (%s and %s) \n", filespec, trc, edf);
	tr->fd = Ttf_OpenFileForInput(trc, edf);
	if (!tr->fd) 
	{dprintf("Open failed!\n");
        *fp  = NULL;
        return 1;
    }
	dprintf("Open successful!\n");
	tr->max_types  = MAX_CATEGORIES;
    tr->num_types  = 0;
	tr->types      = (DRAW_Category **) malloc( tr->max_types
                                              * sizeof(DRAW_Category *) );
	tr->prime      = NULL;
	tr->ymap       = NULL;
	tr->event_count = 0;
	tr->arrow_count = 0;
	*fp            = tr;
	
  dprintf("About to set up callbacks!\n");
  curcolor = 0;

  /* Try to locate all relevant data in a single pass. */
  firstpass.UserData = *fp;
  firstpass.DefThread = DefThread;
  firstpass.EndTrace = EndTrace;
  firstpass.DefClkPeriod = ClockPeriod;
  firstpass.DefStateGroup = 0;
  firstpass.DefState = DefState;
  firstpass.SendMessage = RegMessage;
  firstpass.RecvMessage = 0;
  firstpass.DefUserEvent = DefUserEvent;
  firstpass.EventTrigger = 0;
  firstpass.EnterState = 0;
  firstpass.LeaveState = 0;
  
  

   
   do {
	recs_read = Ttf_ReadNumEvents((*fp)->fd,firstpass, 1024);
   }
  while ((recs_read >=0 && EndOfTrace == 0)); 
  
   Ttf_CloseFile(tr->fd);

  /* Re-open it for input */
  tr->fd = Ttf_OpenFileForInput(trc, edf);
  
  numstops = countstops;
  thispeak = 0;
  currentkind = -1;
  tot_types = (*fp)->num_types;
  
  if(multiThreaded)
  {
  	int i;
  	int numnodes = maxnode;
	ycordsets++;
	
	offset = (int *)malloc((numnodes+1)*sizeof(int));
	
	offset[0] = 0;
	for(i = 0; i<=maxnode; i++)
	{
		offset[i+1] = offset[i] + countthreads[i];
	}
  }
  
  cb.UserData = *fp;
  cb.DefClkPeriod = 0;
  cb.DefThread = 0;
  cb.DefStateGroup = 0;
  cb.DefState = 0;
  cb.DefUserEvent = 0;
  cb.EventTrigger = EventTrigger;
  cb.EndTrace = EndTrace;
  cb.EnterState = EnterState;
  cb.LeaveState = LeaveState;
  cb.SendMessage = SendMessage;
  cb.RecvMessage = RecvMessage;

  
  dprintf("Callbacks set!\n");
  
  return 0;
}

/****************************************************************************@
*  TRACE_Close - Close a trace file
*
*  Input/Output Parameter:
*. fp - Pointer to a trace file handle
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
*
*  Notes: 
*  The pointer 'fp' is set to NULL on a successful close.
***************************************************************************@*/
TRACE_EXPORT
int TRACE_Close( TRACE_file *fp )
{
    TRACE_file     tr;
    int            idx;

    tr             = *fp;
    if ( tr->types != NULL ) {
        for ( idx = 0; idx < tr->num_types; idx++ ) {
             Category_free( tr->types[ idx ] );
             tr->types[ idx ] = NULL;
        }
        tr->num_types = 0;
        free( tr->types );
    }
    if ( tr->prime != NULL ) {
        Primitive_free( tr->prime );
        tr->prime = NULL;
    }
        if ( tr->fd != NULL )
       Ttf_CloseFile( tr->fd );
    *fp = NULL;
	dprintf("All done!");
    return 0;
}


/*@***************************************************************************
*  TRACE_Peek_next_kind - Determine the kind of the next record
*
*  Input Parameter:
*. fp - Trace file handle
*
*  Output Parameters:
*. next_kind - Type of next record.  The kind 'TRACE_EOF', 
*  which has the value '0', is returned at end-of-file.
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
* in TRACE_Get_err_string() for possible error string.
****************************************************************************@*/
  TRACE_EXPORT
int TRACE_Peek_next_kind( const TRACE_file fp, TRACE_Rec_Kind_t *next_kind )
{

	int recs_read = 1;
	if(fp->num_types != 0)
	{
		dprintf("%d categories to go.\n", fp->num_types);
		*next_kind = TRACE_CATEGORY;
		thispeak = 0;
		return 0;
	}
		
	if(ycordsets > 0 && currentkind == (int)TRACE_EOF)
	{
		*next_kind = TRACE_YCOORDMAP;
		ycordsets--;
		return 0;
	}
	if(currentkind == (int)TRACE_EOF)
	{
		*next_kind = (TRACE_Rec_Kind_t)currentkind;

		return 0;
	}
	
	thispeak = 0;
	
	while(thispeak == 0 && recs_read > 0)
	{
		
		recs_read = Ttf_ReadNumEvents(fp->fd,cb, 1);


	}
	thispeak = 0;
	

	if(ycordsets > 0 && currentkind == (int)TRACE_EOF)
	{
		*next_kind = TRACE_YCOORDMAP;
		ycordsets--;
		return 0;
	}
	
	*next_kind = (TRACE_Rec_Kind_t)currentkind;

	return 0;
}

/* Once the kind of the next item is determined, one of the next 4
   routines may be called */
/****************************************************************************@
*  TRACE_Get_next_method - Get the next method description
*
*  Input Parameter:
*. fp - Trace file handle
*
*  Output Parameters:
*+ method_name - 
*. method_extra - 
*- method_id - 
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
****************************************************************************@ */
  TRACE_EXPORT
int TRACE_Get_next_method( const TRACE_file fp,
                           char method_name[], char method_extra[], 
                           int *method_id )
{dprintf("GNM"); return TRACEINPUT_FAIL;}

/****************************************************************************@
*  TRACE_Peek_next_category - Peek at the next category to determine necessary 
*  data sizes
*
*  Input Parameter:
*. fp - Trace file handle
*
*  Output Parameters:
*+ n_legend - Number of characters needed for the legend
*. n_label - Number of characters needed for the label
*- n_methodIDs - Number of methods (Always zero or one in this version) 
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
****************************************************************************@*/
  TRACE_EXPORT
int TRACE_Peek_next_category( const TRACE_file fp,
                              int *n_legend, int *n_label, 
                              int *n_methodIDs )
{


	*n_methodIDs = 0;
	
	if(fp->types[(fp->num_types)-1]->label !=  NULL)
	{   dprintf("PNC: %s",fp->types[(fp->num_types)-1]->label);
		*n_label = strlen(fp->types[(fp->num_types)-1]->label)+1;
	}
	
	*n_legend = strlen(fp->types[(fp->num_types)-1]->legend)+1;
/*
	dprintf("PNC: %s",fp->types[(fp->num_types)-1]->legend);
	if(fp->types[(fp->num_types)-1]->legend !=  NULL)
	{   dprintf("PNC: %s",fp->types[(fp->num_types)-1]->legend);
		*n_legend = strlen(fp->types[(fp->num_types)-1]->legend)+1;
	}
	dprintf("PNC: %d, %d\n", *n_legend, *n_label);*/
	return 0;
}

/*@***************************************************************************
*  TRACE_Get_next_category - Get the next category description
*
*  Input Parameter:
*+ fp - Trace file handle
*. legend_max - Allocated size of 'legend_base' array
*. label_max - Allocated size of 'label_base' array
*- methodID_max - Allocated size of 'methodID_base' array
*
*  Input/Output Parameters:
*+ legend_pos - On input, the first available position in 'legend_base'
*  On output, changed to indicate the new first available position.
*. label_pos - On input, the first available position in 'label_base'
*  On output, changed to indicate the new first available position.
*- methodID_pos - On input, the first available position in 'methodID_base'
*  On output, changed to indicate the new first available position.
* 
*  Output Parameters:
*+ head - Contains basic category info (see the description of 
*  'TRACE_Category_head_t')
*. n_legend - Size of 'legend_base' array to be used
*. legend_base - Pointer to storage to hold legend information
*. n_label - Size of 'label_base' array to be used
*. label_base - Pointer to storage to hold label information.
*               The order of the % tokens specified here, 'label_base', 
*               must match the order of operands in the byte array,
*               'byte_base[]', specified in 'TRACE_Get_next_primitive()'
*               and 'TRACE_Get_next_composite()'.
*. n_methodIDs - number of method IDs associated with this category.
*- methodID_base - Pointer to storage to hold method IDs.
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
****************************************************************************@*/
  TRACE_EXPORT
int TRACE_Get_next_category( const TRACE_file fp,
                             TRACE_Category_head_t *head,
                             int *num_legend, char legend_base[],
                             int *legend_pos, const int legend_max,
                             int *num_label, char label_base[],
                             int *label_pos, const int label_max,
                             int *num_methods, int method_base[],
                             int *method_pos, const int method_max )
{
	
	
	DRAW_Category  *type;
    int             legend_len, label_len;

    type  = fp->types[ fp->num_types-1 ];
    if ( type == NULL ) {
      /* fprintf( stderr, "TRACE_Get_next_category(): Cannot locate "
	 "current category in Category Table.\n" );*/
        return 20;
    }
   

    /* Copy current Category_head_t to the caller's allocated buffer */
    Category_head_copy( head, type->hdr );

    if ( type->legend != NULL ) {
        legend_len = strlen( type->legend );
        if ( legend_len > 0 ) {
            if ( *legend_pos >= legend_max )
                return 21;
            memcpy( &(legend_base[ *legend_pos ]), type->legend,
                    sizeof( char ) * legend_len );
            *num_legend  = legend_len;
            *legend_pos += *num_legend;
            if ( *legend_pos > legend_max )
                return 22;
        }
    }
	
	if ( type->label != NULL ) 
	{
		
        label_len = strlen( type->label );
        if ( label_len > 0 ) 
		{
            if ( *label_pos >= label_max )
                return 23;
            memcpy( &(label_base[ *label_pos ]), type->label,
                    sizeof( char ) * label_len );
            *num_label  = label_len;
            *label_pos += *num_label;
			dprintf("GNC: %d, %s\n", label_len, type->label);
            if ( *label_pos > label_max )
                return 24;
        }
    }


	(fp->num_types)--;
	return 0;
}


/*@***************************************************************************
*  TRACE_Peek_next_primitive - Peek at the next primitive drawable to 
*  determine necessary data sizes and time range
*
*  Input Parameter:
*. fp - Trace file handle
*
*  Output Parameters:
*+ starttime, endtime - time range for drawable
*. nt_coords - Number of time coordinates
*. ny_coords - Number of y coordinates
*- n_bytes - Number of data bytes
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
*
*  Notes:
*  This function really serves two purposes.
*  The time range allows the SLOG2 algorithm to determine which treenode a
*  drawable should be placed in (which may influence where in memory the data 
*  is read by 'TRACE_Get_next_primitive()').  
*  The other return values allow the calling code to allocate space for the 
*  variable-length data in a drawable before calling 'TRACE_Get_next_primitive'.
*
***************************************************************************@*/
  TRACE_EXPORT
int TRACE_Peek_next_primitive( const TRACE_file fp,
                               double *starttime, double *endtime,
                               int *nt_coords, int *ny_coords, int *n_bytes )
{   
	dprintf("PNP\n"); 

	*starttime = fp->prime->starttime;
	*endtime = fp->prime->endtime;
	*nt_coords = *ny_coords = 2;
	*n_bytes = fp->prime->num_info;

	return 0;
}

/*@***************************************************************************
*  TRACE_Get_next_primitive - Get the next primitive drawable
*
*  Input Parameter:
*+ fp - Trace file handle
*. tcoord_max - Size of 'tcoord_base'
*. ycoord_max - Size of 'ycoord_base'
*- byte_max - Size of 'byte_base'
*
*  Input/Output Parameters:
*+ tcoord_pos - On input, the first free location in 'tcoord_base'.  Updated
*               on output to the new first free location.
*. ycoord_pos - The same, for 'ycoord_base'
*- byte_pos -  The same, for 'byte_base'
*
*  Output Parameters:
*+ starttime, endtime - time range for drawable
*. category_index - Index of the category that this drawable belongs to
*. nt_coords - Number of time coordinates
*. tcoord_base - Pointer to storage to hold time coordinates
*. ny_coords - Number of y coordinates
*. ycoord_base - Pointer to storage to hold y coordinates
*- byte_base - Pointer to storage to hold bytes.  The order of operands
*              in the byte array, 'byte_base[]', specified here must match
*              the order of the % tokens in the label string, 'label_base',
*              in the TRACE_Get_next_category().
*
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
***************************************************************************@*/
  TRACE_EXPORT
int TRACE_Get_next_primitive( const TRACE_file fp, 
                              int *category_index, 
                              int *num_tcoords, double tcoord_base[],
                              int *tcoord_pos, const int tcoord_max, 
                              int *num_ycoords, int ycoord_base[], 
                              int *ycoord_pos, const int ycoord_max,
                              int *num_bytes, char byte_base[],
                              int *byte_pos, const int byte_max )
{
	
	DRAW_Primitive *prime;
	int primesize;
	dprintf("GNP\n");  



    if ( fp->prime == NULL ) {
      /*fprintf( stderr, "TRACE_Get_next_primitive(): Cannot locate "
	"Primitive in TRACE_file.\n" );*/
        return 30;
    }
    prime = fp->prime;
    *category_index = prime->type_idx;
	primesize = sizeof(prime->info);
    if ( prime->num_info > 0 ) {dprintf("pni %d, %d, %d, %d\n", prime->num_info, prime->info[0], *byte_pos, primesize);
        if (*byte_pos >= byte_max )
            return 31;

        memcpy( &(byte_base[ *byte_pos ]), prime->info,
                sizeof( char ) * prime->num_info );
        *num_bytes = prime->num_info;
        *byte_pos += *num_bytes;
        if ( *byte_pos > byte_max )
            return 32;
    }

    if ( *tcoord_pos >= tcoord_max )
        return 33;
    memcpy( &(tcoord_base[ *tcoord_pos ]), prime->tcoords,
            sizeof( double ) * prime->num_tcoords );
    *num_tcoords = prime->num_tcoords;
    *tcoord_pos += *num_tcoords;
    if ( *tcoord_pos > tcoord_max )
        return 34;

    if ( *ycoord_pos >= ycoord_max )
        return 35;
    memcpy( &(ycoord_base[ *ycoord_pos ]), prime->ycoords,
            sizeof( int ) * prime->num_ycoords );
    *num_ycoords = prime->num_ycoords;
    *ycoord_pos += *num_ycoords;
    if ( *ycoord_pos > ycoord_max )
        return 36;

    return 0;
}

/*@***************************************************************************
  TRACE_Peek_next_composite - Peek at the next composite drawable to
  determine the number of primitive drawables in this composite object,
  time range, and size of pop up data.

  Input Parameter:
. fp - Trace file handle

  Output Parameters:
+ starttime, endtime - time range for drawable
. n_primitives - Number of primitive drawables in this composite object.
- n_bytes - Number of data bytes

  Return Value:
. ierr - Returned integer error status.  It will be used as an argument
  in TRACE_Get_err_string() for possible error string.
***************************************************************************@*/
  TRACE_EXPORT
int TRACE_Peek_next_composite( const TRACE_file fp,
                               double *starttime, double *endtime,
                               int *n_primitives, int *n_bytes )
{dprintf("PNCo"); return TRACEINPUT_FAIL;}

/*@***************************************************************************
  TRACE_Get_next_composite - Get the header information of the 
                             next composite drawable

  Input Parameter:
+ fp - Trace file handle
- byte_max - Size of 'byte_base'

  Input/Output Parameters:
. byte_pos -  The same, for 'byte_base'

  Output Parameters:
+ starttime, endtime - time range for drawable
. category_index - Index of the category that this drawable belongs to
- byte_base - Pointer to storage to hold bytes.  The order of operands
              in the byte array, 'byte_base[]', specified here must match
              the order of the % tokens in the label string, 'label_base',
              in the TRACE_Get_next_category().


  Return Value:
. ierr - Returned integer error status.  It will be used as an argument
  in TRACE_Get_err_string() for possible error string.

  Notes:
  The interface to this is designed to allow flexibility in how data is read.
  See 'TRACE_Get_next_primitive' for more details.
  ***************************************************************************@*/
  TRACE_EXPORT
int TRACE_Get_next_composite( const TRACE_file fp,
                              int *category_index,
                              int *n_bytes, char byte_base[],
                              int *byte_pos, const int byte_max )
{dprintf("PNCo"); return TRACEINPUT_FAIL;}


/*@***************************************************************************
*  TRACE_Get_position - Return the current position in an trace file
*
*  Input Parameter:
*. fp - Trace file handle
*
*  Output Parameter:
*. offset - Current file offset.
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
*
*
*  Notes:
*  This routine and 'TRACE_Set_position' are used in the construction of an
*  annotated Slog file.  In an annotated Slog file, the Slog file records the
*  location in the original trace file of the records, rather than making a
*  copy of the records.  
*
*  If the trace file is actually a collection of files, then that information
*  should be encoded within the position.  
***************************************************************************@*/
  TRACE_EXPORT
int TRACE_Get_position( TRACE_file fp, TRACE_int64_t *offset )
{dprintf("GNPo");  return TRACEINPUT_FAIL;}

/*@***************************************************************************
  TRACE_Set_position - Set the current position of a trace file

  Input Parameters:
+ fp - Trace file handle
- offset - Position to set file at

  Return Value:
. ierr - Returned integer error status.  It will be used as an argument
  in TRACE_Get_err_string() for possible error string.

  Notes:
  The file refered to here is relative to the 'filespec' given in a 
  'TRACE_Open' call.  If that 'filespec' describes a collection of real files,
  then this calls sets the position to the correct location in the correct
  real file.
***************************************************************************@*/
TRACE_EXPORT
int TRACE_Set_position( TRACE_file fp, TRACE_int64_t offset )
{dprintf("SPo");return TRACEINPUT_FAIL;}

/*@***************************************************************************
  TRACE_Get_err_string - Return the error string corresponding to an error code

  Input Parameter:
. ierr - Error code returned by a TRACE routine

  Return Value:
  Error message string.

  Notes:
  This routine is responsible for providing internationalized (translated)
  error strings.  Implementors may want to consider the GNU 'gettext' style
  functions.  To avoid returning messages of the form 'Message catalog not 
  found', the message catalog routines such as 'catopen' and 'catgets' should
  not be used unless a provision is made to return a message string if no
  message catalog can be found.   The help message for the TRACE-API 
  implementation should be stored at ierr=0, so the calling program of
  TRACE-API knows if it should exit the program normally.
***************************************************************************@*/
TRACE_EXPORT
char *TRACE_Get_err_string( int ierr )
{
    switch ( ierr ) {
        case 0:
            return "Usage: executable_name ASCII_drawable_filename";
        case 1:
            return "Error: fopen() fails!";
        case 10:
            return "Maximum of Categories has been reached.";
        case 20:
            return "Cannot locate CATEGORY in the internal table.";
        case 21:
            return "TRACE_Get_next_category(): Memory violation "
                   "detected before writing Legend.\n";
        case 22:
            return "TRACE_Get_next_category(): Memory violation "
                   "detected after writing Legend.\n";
        case 23:
            return "TRACE_Get_next_category(): Memory violation "
                   "detected before writing Label.\n";
        case 24:
            return "TRACE_Get_next_category(): Memory violation "
                   "detected after writing Label.\n";
        case 25:
            return "TRACE_Get_next_category(): Memory violation "
                   "detected before writing MethodIDs.\n";
        case 26:
            return "TRACE_Get_next_category(): Memory violation "
                   "detected after writing MethodIDs.\n";
        case 30:
            return "Cannot locate PRIMITIVE in the internal table.";
        case 31:
            return "TRACE_Get_next_primitive(): Memory violation "
                   "detected before writing ByteInfo.\n";
        case 32:
            return "TRACE_Get_next_primitive(): Memory violation "
                   "detected after writing ByteInfo.\n";
        case 33:
            return "TRACE_Get_next_primitive(): Memory violation "
                   "detected before writing Time coordinates.\n";
        case 34:
            return "TRACE_Get_next_primitive(): Memory violation "
                   "detected after writing Time coordinates.\n";
        case 35:
            return "TRACE_Get_next_primitive(): Memory violation "
                   "detected before writing Yaxis coordinates.\n";
        case 36:
            return "TRACE_Get_next_primitive(): Memory violation "
                   "detected after writing Yaxis coordinates.\n";
        case 40:
            return "Cannot locate COMPOSITE in the internal table.";
        case 41:
            return "TRACE_Get_next_composite(): Memory violation "
                   "detected before writing ByteInfo.\n";
        case 42:
            return "TRACE_Get_next_composite(): Memory violation "
                   "detected after writing ByteInfo.\n";
        case 49:
            return "TRACE_Peek_next_composite(): Unexpected EOF detected.";
        case 60:
            return "Cannot locate YCOORDMAP in the internal table.";
        case 61:
            return "TRACE_Peek_next_ycoordmap(): Inconsistency detected "
                   "in the number of methods from input text file.\n";
        case 63:
            return "TRACE_Get_next_ycoordmap(): Memory violation "
                   "detected before writing Yaxis coordinate map.\n";
        case 64:
            return "TRACE_Get_next_ycoordmap(): Memory violation "
                   "detected after writing Yaxis coordinate map.\n";
        case 65:
            return "TRACE_Get_next_ycoordmap(): Memory violation "
                   "detected before writing MethodIDs.\n";
        case 66:
            return "TRACE_Get_next_ycoordmap(): Memory violation "
                   "detected after writing Methods.\n";
        default:
            return "Unknown Message ID ";
    }
}


/* 
 * The following allow the input api to specify how to identify the
 * y-axis coordinates
 */
/*@***************************************************************************
*  TRACE_Peek_next_ycoordmap - Get the size and the description
*  of the y-axis coordinate map
*
*  Input Parameter:
*. fp - Pointer to a trace file handle
*
*  Output Parameters:
*+ n_rows - Number of rows of the y-axis coordinate map
*. n_columns - Number of columns of the Yaxis coordinate map
*. max_column_name - The maximum length of the column name arrays, i.e.
*                    max_column_name = MAX( { column_name[i] } )
*. max_title_name - Title string for this map
*- n_methodIDs - Number of Method IDs associated with this map
*
*  Return Value:
*. ierr - Returned integer error status.  It will be used as an argument
*  in TRACE_Get_err_string() for possible error string.
*
*  Notes:
*  Both 'max_column_name' and 'max_title_name' includes the NULL character
*  needed at the end of the 'title_name' and 'column_names[i]' used in
*  'TRACE_Get_next_ycoordmap()'
***************************************************************************@*/
  
  TRACE_EXPORT
int TRACE_Peek_next_ycoordmap( TRACE_file fp,
                               int *num_rows, int *num_columns,
                               int *max_column_name,
                               int *max_title_name,
                               int *num_methods )
{
	DRAW_YCoordMap *ymap;

    int             Nrows = rows;
	int				Ncols = 3; 
	int				Nmeths = 0;
    int            *map_elems;
    int             max_colnames;
    char            str4methods[MAX_LABEL_LEN];
    int             methods_len;
    int             icol, irow, idx;

 #if defined( DEBUG )
    dprintf( "%s(%d,%d,%d)] :\n", mapname, Nrows, Ncols, Nmeths );
#endif


    ymap = YCoordMap_alloc( Nrows, Ncols, Nmeths );

    
	
	memcpy(ymap->title_name, "Thread View", MAX_NAME_LEN);
	

#if defined( DEBUG )
    printf( "Title=%s \nColumnLabels=< LineID -> ", ymap->title_name );
#endif

    max_colnames = 0;
    for ( icol = 0; icol < Ncols-1; icol++ ) {

		
		if(icol == 0)
		{
			memcpy(ymap->column_names[icol], "NodeID", 7);
		}
		else
		{
			memcpy(ymap->column_names[icol], "ThreadID", 9);
		}

#if defined( DEBUG )
        printf( "%s ", ymap->column_names[icol] );
#endif
        if ( max_colnames < strlen( ymap->column_names[icol] ) + 1 )
            max_colnames = strlen( ymap->column_names[icol] ) + 1;
    }

#if defined( DEBUG )
    printf( ">\n" );
#endif

    map_elems = ymap->elems;
    idx = 0;
	for ( irow = 0; irow <= maxnode; irow++ ) 
	{
		for(icol = 0; icol<=countthreads[irow]; icol++)
		{
			map_elems[idx] =  GlobalID(irow,icol);
			map_elems[idx+1] = irow;
			map_elems[idx+2] = icol;
			idx+=3;
		}
	}	
	
    /*    sscanf( newline, "( %d %n", &map_elems[ idx++ ], &line_pos ); 
        //newline = (char *) (newline+line_pos);
#if defined( DEBUG )
        printf( "%d -> ", map_elems[ idx-1 ] );
#endif
        for ( icol = 1; icol < Ncols-1; icol++ ) {
            sscanf( newline, "%d %n", &map_elems[ idx++ ], &line_pos );
            //newline = (char *) (newline+line_pos);
#if defined( DEBUG )
            printf( "%d ", map_elems[ idx-1] );
#endif
        }
        sscanf( newline, "%d ) %n", &map_elems[ idx++ ], &line_pos ); 
        //newline = (char *) (newline+line_pos);
#if defined( DEBUG )
        printf( "%d\n", map_elems[ idx-1 ] );
#endif*/
    

    /* Set Methods 
    info_A = NULL;
    info_B = NULL;
    if (    ( info_A = strstr( newline, "{ " ) ) != NULL
         && ( info_B = strstr( info_A, " }" ) ) != NULL ) {
        info_A = (char *) (info_A + 2);
        sprintf( info_B, "%c", '\0' );
        strncpy( str4methods, info_A, MAX_LABEL_LEN );
#if defined( DEBUG )
        printf( "{%s}", str4methods );
#endif
        newline = (char *) (info_B + 2);
        // Assume only 1 method ID 
        methods_len = 1;
    }
    else*/
        methods_len = 0;
#if defined( DEBUG )
    printf( "\n" );
#endif

    if ( methods_len != Nmeths ) {
      /* fprintf( stderr, "TRACE_Peek_next_ycoordmap(): The number of methods "
                         "defined is %d and number read is %d\n",
                         Nmeths, methods_len );*/
        return 61;
    }

    if ( methods_len > 0 ) {
        /* Assume 1 method ID */
        ymap->methods[ 0 ] = atoi( str4methods );
    }
	
    *num_rows         = ymap->num_rows;
    *num_columns      = ymap->num_columns;
    *max_column_name  = max_colnames;
    *max_title_name   = strlen( ymap->title_name ) + 1;
    *num_methods      = methods_len;

    /* Free the previous allocated YCoordMap stored at TRACE_file */
    YCoordMap_free( fp->ymap );

    fp->ymap = ymap;

    return 0;
}

/*@***************************************************************************
  TRACE_Get_next_ycoordmap - Return the content of a y-axis coordinate map 


  Output Parameters:

  Input Parameters:
+ fp - Pointer to a trace file handle
. coordmap_max - Allocated size of 'coordmap_base' array
- methodID_max - Allocated size of 'methodID_base' array

  Input/Output Parameters:
+ coordmap_pos - On input, the first free location in 'coordmap_base'.
                 Updated on output to the new first free location.
- methodID_pos - On input, the first available position in 'methodID_base'
                 On output, changed to indicate the new first available
                 position.

  Output Parameters:
+ title_name - character array of length, 'max_title_name', is assumed 
               on input, where 'max_title_name' is defined by 
               'TRACE_Peek_next_ycoordmap()'.  The title name of this 
               map which is NULL terminated will be stored in this
               character array on output.
. column_names - an array of character arrays to store the column names.
                 Each character array is of length of 'max_column_name'.
                 There are 'ncolumns-1' character arrays altogether.
                 where 'ncolumns' and 'max_column_name' are returned by 
                 'TRACE_Peek_next_ycoordmap()'.  The name for the first 
                 column is assumed to be known, only the last 'ncolumns-1' 
                 columns need to be labeled.
. coordmap_sz - Total number of integers in 'coordmap[][]'.
                'coordmap_sz' = 'nrows' * 'ncolumns', 
                otherwise an error will be flagged.
                Where 'nrows' and 'ncolumns' are returned by
                'TRACE_Peek_next_ycoordmap()'
. coordmap_base - Pointer to storage to hold y-axis coordinate map.
. n_methodIDs - number of method IDs associated with this map.
- methodID_base - Pointer to storage to hold method IDs.


  Return Value:
. ierr - Returned integer error status.  It will be used as an argument
  in TRACE_Get_err_string() for possible error string.

  Notes:
  Each entry in y-axis coordinate map is assumed to be __continuously__ 
  stored in 'coordmap_base[]', i.e. every 'ncolumns' consecutive integers 
  in 'coordmap_base[]' is considered one coordmap entry.
  ***************************************************************************@*/

  TRACE_EXPORT
int TRACE_Get_next_ycoordmap( TRACE_file fp,
                              char *title_name,
                              char **column_names,
                              int *coordmap_sz, int coordmap_base[],
                              int *coordmap_pos, const int coordmap_max,
                              int *num_methods, int method_base[],
                              int *method_pos, const int method_max )
{
	int              icol;
	DRAW_YCoordMap  *ymap;
	
	dprintf("GNY"); 
    

    if ( fp->ymap == NULL ) {
      /*fprintf( stderr, "TRACE_Get_next_ycoordmap(): Cannot locate "
	"YCoordMap in TRACE_file.\n" );*/
        return 60;
    }
    ymap = fp->ymap;

    strcpy( title_name, ymap->title_name );
    /*
    fprintf( stderr, "strlen(%s) = %d\n", title_name, strlen(title_name) );
    fflush( stderr );
    */
    for ( icol = 0; icol < ymap->num_columns - 1; icol++ ) {
        /*
        fprintf( stderr, "strlen(%s) = %d\n", ymap->column_names[icol],
                         strlen(ymap->column_names[icol]) );
        fflush( stderr );
        */
        strcpy( column_names[icol], ymap->column_names[icol] );
    }

    if ( *coordmap_pos >= coordmap_max )
        return 63;
    memcpy( &(coordmap_base[ *coordmap_pos ]), ymap->elems,
            sizeof( int ) * ymap->num_rows * ymap->num_columns );
    *coordmap_sz   = ymap->num_rows * ymap->num_columns;
    *coordmap_pos += *coordmap_sz;
    if ( *coordmap_pos > coordmap_max )
        return 64;

    if ( ymap->num_methods > 0 ) {
        if ( *method_pos >= method_max )
            return 65;
        memcpy( &(method_base[ *method_pos ]), ymap->methods,
                sizeof( int ) * ymap->num_methods );
        *num_methods = ymap->num_methods;
        *method_pos += *num_methods;
        if ( *method_pos > method_max )
            return 66;
    }

    return 0;
}
