#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TRIAL* perfdmf_query_trials(PGconn* connection, PERFDMF_EXPERIMENT* experiment) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_trials(%p)\n", experiment);
#endif
  char my_query[256];
  sprintf(my_query,"DECLARE myportal CURSOR FOR select * from trial where experiment = %d", experiment->id);

  return taudb_private_query_trials(connection, FALSE, my_query);
}

