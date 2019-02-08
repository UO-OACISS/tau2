/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2014
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */
/****************************************************************************
**  SCALASCA    http://www.scalasca.org/                                   **
**  KOJAK       http://www.fz-juelich.de/jsc/kojak/                        **
*****************************************************************************
**  Copyright (c) 1998-2009                                                **
**  Forschungszentrum Juelich, Juelich Supercomputing Centre               **
**                                                                         **
**  See the file COPYRIGHT in the package base directory for details       **
****************************************************************************/

#ifndef OPARI2_CTC_TOKEN_H
#define OPARI2_CTC_TOKEN_H

#include "pomp2_region_info.h"
#include "pomp2_user_region_info.h"

/** CTC Tokens */
typedef enum
{
    CTC_No_token = 0,
    CTC_REGION_TOKENS,
    CTC_OMP_TOKENS,
    CTC_USER_REGION_TOKENS
} CTCToken;

#endif /* OPARI2_CTC_TOKEN_H */
