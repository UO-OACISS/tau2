/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1993,1995             */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

#   ifdef _CRAYT3E
#     define _havehosttype_
ARCHt3e
#else
#   ifdef CRAY-C90            /* CRAYTEST */
#     define _havehosttype_   /* CRAYTEST */
ARCHc90                       /* CRAYTEST */
#   else                      /* CRAYTEST */
#     define _havehosttype_   /* CRAYTEST */
ARCHcray                      /* CRAYTEST */
#   endif /* CRAY-C90 */      /* CRAYTEST */
#   endif
