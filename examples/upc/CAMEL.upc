#include <upc_relaxed.h>

/* CAMEL cipher differential cryptanalysis
 *
 * Written by:    Chris Conger and Matt Murphy
 *           11/2003
 *
 * Last Update:   2/13/2003
 *
 * Conversion to UPC by: Adam Leko, ca Summer 2004
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLUMN unsigned int
#define ROW unsigned int
#define ELEMENT unsigned int
#define INDEX unsigned int
#define UINT unsigned int
#define WORDSIZE 32
#define KEYSIZE 32
#define SPLITKEYSIZE 16
#define SUBKEYSIZE WORDSIZE
#define BLOCKSIZE SUBKEYSIZE
#define SPLITSIZE 16

#ifndef NUMPAIRS
#define NUMPAIRS 200000
#endif

#ifndef MAINKEYLOOP
#define MAINKEYLOOP 64
#endif

void extendDC(UINT* R1XCHAR, UINT* R1YCHAR, int dex[], int dey[]);
void quicklySort(UINT max);
UINT hex2UINT(char *s);
void UINT2hex(UINT ky, char cResult[8]);
ELEMENT SIO(COLUMN col,UINT S[256]);
UINT sortNcount(UINT max, UINT keyCount[2][4096], UINT* highCount);
int docrypt(UINT key32, UINT inputbits, UINT whichone, UINT* R1X, UINT* R1Y, UINT* C, UINT* C2);
UINT lastRound(UINT inputbits, UINT key);

/*================================Global Variables============================*/
UINT E[] = {15,0,1,2,1,2,3,4,3,4,5,6,5,6,7,8,7,8,9,10,9,10,11,12,11,12,13,14,13,14,15,0};
UINT p1[] = {7,15,23,31,6,14,22,30,5,13,21,29,4,12,20,28,3,11,19,27,2,10,18,26,1,9,17,25,0,8,16,24};
UINT p2[] = {1,2,5,6,9,10,13,14,17,18,21,22,25,26,29,30,0,3,4,7,8,11,12,15,16,19,20,23,24,27,28,31};
UINT p3[] = {11,6,0,8,15,7,4,1,13,5,14,9,2,10,3,12};
UINT pInverse[] = {12,10,8,14,4,2,0,6,13,11,9,15,5,3,1,7,29,27,25,31,21,19,17,23,30,28,26,24,22,20,18,16};

UINT s[4][256];
UINT s0[] = {2,6,3,12,7,9,1,15,10,0,4,8,14,5,11,13,
  4,9,0,1,3,10,15,11,8,2,6,12,5,13,7,14,
  11,5,10,9,4,12,7,1,13,3,0,15,2,8,14,6,
  7,1,15,2,0,8,12,5,14,6,3,13,9,10,11,4,
  0,7,14,4,9,12,1,8,3,6,11,13,2,10,15,5,
  3,10,8,2,6,14,0,9,11,4,13,1,15,12,5,7,
  6,14,5,10,9,0,13,7,4,15,8,3,12,1,2,11,
  6,10,1,9,11,5,4,8,12,14,2,0,13,15,7,3,
  12,3,9,5,13,1,11,4,2,7,14,6,10,0,15,8,
  3,6,13,2,15,8,14,1,5,12,4,11,7,9,0,10,
  9,0,5,14,1,4,10,13,6,8,12,2,11,7,3,15,
  1,4,11,3,10,13,2,6,12,9,5,7,15,14,8,0,
  5,12,4,0,8,11,6,9,1,10,7,14,3,2,13,15,
  5,11,2,9,3,13,10,0,6,1,14,8,12,4,15,7,
  8,10,13,7,5,14,3,11,0,15,9,4,12,6,1,2,
  15,12,4,6,14,3,8,2,7,5,11,9,0,13,10,1};

UINT s1[] = {8,1,4,14,9,5,0,3,13,7,10,2,15,12,6,11,
  0,9,5,8,2,10,13,1,7,4,14,6,11,3,15,12,
  1,6,14,5,11,15,2,10,12,0,13,3,8,7,9,4,
  9,8,1,15,0,14,7,2,10,11,3,12,4,6,5,13,
  2,4,10,11,15,3,12,5,0,9,6,8,1,13,14,7,
  3,5,15,9,12,2,10,4,6,1,8,13,0,11,7,14,
  11,12,2,1,7,6,14,8,3,13,4,15,5,9,10,0,
  9,0,14,10,1,7,3,2,11,4,15,8,12,5,13,6,
  4,15,14,12,13,5,11,6,9,10,7,0,3,8,2,1,
  12,3,4,0,14,11,5,9,13,2,15,6,7,1,8,10,
  6,7,10,11,5,8,15,13,12,14,0,1,9,2,4,3,
  15,6,0,14,3,12,13,7,2,5,9,4,11,10,1,8,
  7,12,13,3,15,4,1,11,14,8,10,5,6,9,0,2,
  5,2,3,4,10,1,13,0,8,15,6,12,11,14,7,9,
  14,11,8,7,4,0,9,5,13,3,1,10,2,12,6,15,
  13,10,7,2,8,15,4,9,1,6,11,14,12,0,3,5};

UINT s2[] = {6,8,4,0,9,6,1,7,2,10,15,6,13,14,11,3,
  12,2,3,4,10,11,9,0,7,8,2,4,1,4,14,6,
  1,5,10,3,5,1,4,12,11,15,0,7,5,7,13,11,
  2,15,9,11,3,0,8,10,6,9,1,1,11,8,8,0,
  9,1,11,12,4,15,7,4,9,2,3,10,8,3,5,10,
  3,4,7,6,13,5,3,5,1,1,8,14,4,13,2,5,
  8,10,1,15,8,14,6,6,8,6,9,3,15,5,9,13,
  11,12,6,5,12,10,2,9,15,5,14,13,7,10,6,1,
  15,14,12,10,7,4,14,13,13,14,5,9,0,9,0,2,
  14,9,0,9,15,9,11,15,12,4,4,11,12,1,1,15,
  0,3,13,13,1,3,15,11,14,0,6,12,14,0,3,8,
  4,6,15,8,2,13,13,14,5,13,10,0,3,12,7,14,
  13,11,2,2,11,2,5,3,0,3,7,15,10,2,4,7,
  10,0,14,14,14,8,10,1,4,7,13,5,2,6,12,12,
  7,13,8,7,0,12,12,2,10,12,11,2,6,11,13,9,
  5,7,5,1,6,7,0,8,3,11,12,8,9,15,10,4};

UINT s3[] = {7,10,14,5,0,8,4,11,3,15,1,2,12,14,6,9,
  4,0,2,9,12,13,14,8,11,7,10,3,4,5,9,6,
  11,13,15,3,2,7,8,12,0,6,4,9,14,9,12,1,
  13,9,5,8,4,9,3,2,8,11,7,0,13,10,1,14,
  1,7,9,12,13,14,15,0,5,3,6,8,3,4,10,11,
  14,3,13,1,5,15,9,4,12,0,3,7,11,15,7,10,
  9,12,4,15,14,5,5,3,7,1,12,6,9,11,0,13,
  3,15,6,7,9,1,12,14,15,2,11,10,15,6,4,5,
  0,8,11,4,10,3,6,7,9,12,15,1,10,8,13,2,
  12,2,8,6,11,2,0,13,10,5,9,15,1,7,3,4,
  15,14,1,13,6,0,11,9,2,8,0,13,5,2,11,3,
  6,5,12,0,8,4,10,1,6,13,13,11,2,3,14,7,
  5,6,10,2,7,12,13,10,1,4,8,14,0,1,8,15,
  10,4,0,11,15,10,1,6,14,9,14,5,7,13,2,12,
  8,11,3,10,1,6,7,15,13,14,2,4,8,12,5,0,
  2,1,7,14,3,11,2,5,4,10,5,12,6,0,15,8};

static inline unsigned long long timer_now() {
  struct timeval st;
  gettimeofday(&st, NULL);
  return (st.tv_sec * 1e6 + st.tv_usec);
}

static inline double timer_elapsed(unsigned long long start) {
  return (timer_now() - start) * 1e-6;
}

static inline double timer_elapsed2(unsigned long long start, unsigned long long end) {
  return (end - start) * 1e-6;
}

struct cryptresult {
  unsigned int r1y;
  unsigned int curr1y;
  unsigned int c;
  unsigned int c2;
  unsigned int indexfound;
};

// these are separated out from the .h file because they are defined differently
// in each version

#define KEYARRAYSIZE 32768

#define MAXCRESULT 32768

shared unsigned int key32_in;
shared int cresultcnt;
shared int sharedindex;
shared int shared_n;
// block these guys totally on the master thread
// they are not that big, and it makes the sorts go much faster
shared [] struct cryptresult cresult[MAXCRESULT];
shared [] unsigned int PK2[KEYARRAYSIZE];
// everyone uses this, so we better share it
shared unsigned int keyArray[KEYARRAYSIZE * THREADS];

/**********************************FUNCTIONS***********************************/

// function to convert a character-hex number into unsigned int
UINT hex2UINT(char *s) {
  int Len = strlen(s), d;
  UINT temp, uResult = 0;
  char dummy;

  for (d = 0; d < Len; d++) {
    dummy = s[Len - d - 1];
    if (dummy == '0')
      temp = 0;
    else if (dummy == '1')
      temp = 1;
    else if (dummy == '2')
      temp = 2;
    else if (dummy == '3')
      temp = 3;
    else if (dummy == '4')
      temp = 4;
    else if (dummy == '5')
      temp = 5;
    else if (dummy == '6')
      temp = 6;
    else if (dummy == '7')
      temp = 7;
    else if (dummy == '8')
      temp = 8;
    else if (dummy == '9')
      temp = 9;
    else if (dummy == 'a' || dummy == 'A')
      temp = 10;
    else if (dummy == 'b' || dummy == 'B')
      temp = 11;
    else if (dummy == 'c' || dummy == 'C')
      temp = 12;
    else if (dummy == 'd' || dummy == 'D')
      temp = 13;
    else if (dummy == 'e' || dummy == 'E')
      temp = 14;
    else
      temp = 15;
    uResult |= ((temp & 15) << (d*4));
  }
  return uResult;
}

// function to convert an unsigned integer to a character hexadecimal number
void UINT2hex(UINT ky, char cResult[8]) {
  UINT tmp, counter;

  for (counter = 0;counter < 8;counter++) {
    tmp = ((ky & (15<<(28-4*counter)))>>(28-4*counter));
    if (tmp == 0)
      cResult[counter] = '0';
    else if (tmp == 1)
      cResult[counter] = '1';
    else if (tmp == 2)
      cResult[counter] = '2';
    else if (tmp == 3)
      cResult[counter] = '3';
    else if (tmp == 4)
      cResult[counter] = '4';
    else if (tmp == 5)
      cResult[counter] = '5';
    else if (tmp == 6)
      cResult[counter] = '6';
    else if (tmp == 7)
      cResult[counter] = '7';
    else if (tmp == 8)
      cResult[counter] = '8';
    else if (tmp == 9)
      cResult[counter] = '9';
    else if (tmp == 10)
      cResult[counter] = 'A';
    else if (tmp == 11)
      cResult[counter] = 'B';
    else if (tmp == 12)
      cResult[counter] = 'C';
    else if (tmp == 13)
      cResult[counter] = 'D';
    else if (tmp == 14)
      cResult[counter] = 'E';
    else
      cResult[counter] = 'F';
  }
}

// these functions below are short-lived and expensive to instrument
#pragma pupc off

/************************* CAMEL Encryption Functions **************************
 *
 * Key Schedule functions
 */

// Permutation Box P_1 for key schedule
UINT permut_1(UINT key32) {
  UINT uResult = 0, i = 0;
  for (; i< KEYSIZE; i++)
    if ((1 << (KEYSIZE - p1[i] - 1)) & key32)
      uResult |= 1 << (KEYSIZE - i - 1);
  return uResult;
}

// Permutation Box P_2 for key schedule
UINT permut_2(UINT key16[]) {
  UINT Result=0, temp, i=0;

  temp = (key16[0] << SPLITKEYSIZE) | key16[1];
  for (; i < SUBKEYSIZE; i++)
    if ((1 << (KEYSIZE - p2[i] - 1)) & temp)
      Result |= 1 << (KEYSIZE - i - 1);
  return Result;
}

// Split key into two 16-bit halves
void splitKey(UINT p32Key32, UINT uResult[]) {
  // 4294901760 => 11111111111111110000000000000000
  // 65535 ======> 00000000000000001111111111111111
  UINT leave_upper = 4294901760UL, leave_lower = 65535;
  uResult[0] = (p32Key32 & leave_upper) >> SPLITKEYSIZE;
  uResult[1] = (p32Key32 & leave_lower);
}

// Perform shifts for key schedule
UINT leftShift(UINT nKey, UINT nShift, UINT nSize) {
  UINT n = nKey >> (nSize - nShift), i, nMask = 0;
  nKey <<= nShift;
  for (i = 0; i < nSize; i++)
    nMask |= 1 << i;
  return (nKey | n) & nMask;
}


/** ACTUAL KEY SCHEDULE **/
void keySchedule(UINT key32, UINT round_key32[]) {
  UINT key16[2] = {0, 0}, keyTemp, i;
  keyTemp = permut_1(key32);
  splitKey(keyTemp, key16);

  for (i = 0; i < 2; i++) {
    key16[0] = leftShift(key16[0], i + 1, SPLITKEYSIZE);
    key16[1] = leftShift(key16[1], i + 1, SPLITKEYSIZE);
    round_key32[i] = permut_2(key16);
  }
}


/* ======================== Round ======================= */

void split_block(UINT bInput32, UINT bLR[]) {
  UINT L_mask = 4294901760UL, H_mask = 65535;
  bLR[0] = ((bInput32 & L_mask) >> 16);
  bLR[1] = (bInput32 & H_mask);
}

// f-function
UINT f(UINT bRight, UINT key) {
  UINT bRes = 0, bTemp;
  UINT sLR16[] = {0, 0}, r, c, r2, c2;
  UINT sub_temp[] = {0, 0, 0, 0};
  int i = SUBKEYSIZE;
  while (--i >= 0)
    if ((1 << (16 - E[SUBKEYSIZE - i -1] - 1)) & bRight)
      bRes |= (1 << i);

  bRes ^= key;
  split_block(bRes, sLR16);

  // s0
  c = (sLR16[0] & 15360) >> 10;
  r = ((sLR16[0] & 49152) >> 12) | ((sLR16[0] & 768) >> 8);
  // s1
  c2 = (sLR16[0] & 60) >> 2;
  r2 = ((sLR16[0] & 192) >> 4) | (sLR16[0] & 3);
  sub_temp[0] = s[0][16 * r + c] << 12;
  sub_temp[1] = s[1][16 * r2 + c2] << 8;
  // s2
  c = (sLR16[1] & 15360) >> 10;
  r = ((sLR16[1] & 49152) >> 12) | ((sLR16[1] & 768) >> 8);
  // s3
  c2 = (sLR16[1] & 60) >> 2;
  r2 = ((sLR16[1] & 192) >> 4) | (sLR16[1] & 3);
  sub_temp[2] = s[2][16 * r + c] << 4;
  sub_temp[3] = s[3][16 * r2 + c2];

  bTemp = (sub_temp[0] | sub_temp[1] | sub_temp[2] | sub_temp[3]);
  bRes = 0;

  // perform final permutation (p3) of f-function
  i = 16;
  while (--i >= 0)
    if ((1 << (16 - p3[16 - i - 1] - 1)) & bTemp)
      bRes |= (1 << i);

  return bRes;
}

/* Encrypt given inputbits with 'global key' (inputted by user),
*  spits out round 1 output (R1Y), and two ciphertexts (for X and X^dX)
*/
int docrypt(UINT key32, UINT inputbits, UINT whichone, UINT* R1X, UINT* R1Y, UINT* C, UINT* C2) {
  UINT round_key32[2] = {0, 0};
  UINT input32 = inputbits, exInput32, i;
  UINT LR[] = {0, 0};

  keySchedule(key32, round_key32);
  exInput32 = input32;
  *R1X = exInput32;

  for (i = 0; i < 2; i++) {
    split_block(exInput32, LR);
    input32 = (f(LR[1], round_key32[i]) ^ LR[0]) << 16;
    input32 |= LR[1];
    exInput32 = ((input32 & 4294901760UL) >> 16) | ((input32 & 65535) << 16);
    if (i == 0) {
      *R1Y = exInput32;
    }
  }

  if (whichone == 0) {
    *C = input32;
  } else {
    *C2 = input32;
  }

  return exInput32;
}


/* Encrypt for 1 round only (the last round) given an input and key */
UINT lastRound(UINT inputbits, UINT key) {
  UINT LR[] = {0, 0};
  UINT input32 = inputbits;
  UINT exInput32;
  exInput32 = input32;
  split_block(exInput32, LR);
  input32 = (f(LR[1], key) ^ LR[0]) << 16;
  input32 |= LR[1];
  return input32;
}


/*============================Cryptanalytic Functions=========================*/
/* This function returns the output of the specified SBox 'S', given the
   specified 8-bit input 'col' */
ELEMENT SIO(COLUMN col, UINT S[256]) {
  ELEMENT value;
  UINT r, c;
  c = (col & 60) >> 2;
  r = ((col & 192) >> 4) | (col & 3);
  value = S[16 * r + c];
  return value;
}

UINT sortNcount(UINT max, UINT keyCount[2][4096], UINT* highCount) {
  UINT dmmy, keycounter = 0, q, r;
  UINT highestKC = 0, best = 0;

  // bubble sort entire PK2 array to group numbers together
  for (r = 0; r < max; r++) {
    for (q = 0; q < max; q++)
      if (PK2[q + 1] < PK2[q]) {
        dmmy = PK2[q + 1];
        PK2[q + 1] = PK2[q];
        PK2[q] = dmmy;
      }
  }

  // counter number of each key match, recording both the key and the
  // number of occurences
  dmmy = 0;
  for (r = 0; r < max; r++) {
    if (PK2[r + 1] == PK2[r])
      dmmy++;
    else {
      keyCount[0][keycounter] = PK2[r];
      keyCount[1][keycounter] = dmmy + 1;
      if ((dmmy + 1) >= highestKC) {
        highestKC = dmmy + 1;
        best = keycounter;
      }
      keycounter++;
      dmmy = 0;
    }
  }

  /*  By this point, vars 'highestKC' contains actual highest COUNT encountered,
   *  and 'best' is the INDEX of the keyCount array pointing to correct key;
   *       ==> keyCount[0][best] = Guessed/Broken subkey!
   */
  *highCount = highestKC;

  return keyCount[0][best];
}

void quicklySort(UINT max) {
  UINT q, r, keycounter = 0, dmmy;

  // bubble sort entire PK2 array to group numbers together
  for (r = 0; r < max; r++) {
    for (q = 0; q < max; q++)
      if (PK2[q + 1] < PK2[q]) {
        dmmy = PK2[q + 1];
        PK2[q + 1] = PK2[q];
        PK2[q] = dmmy;
      }
  }

  for (r = 0; r < max; r++)
    if (PK2[r + 1] != PK2[r]) {
      keyArray[keycounter] = PK2[r];
      keycounter++;
    }
}

#pragma pupc on
// end short-lived functions...

void extendDC(UINT* R1XCHAR, UINT* R1YCHAR, int dex[], int dey[]) {
  if ((dex[0] & 128) == 128) {
    *R1XCHAR = (*R1XCHAR | 1);
    *R1YCHAR = (*R1YCHAR | (1 << 16));
  }
  if ((dex[0] & 64) == 64) {
    *R1XCHAR = (*R1XCHAR | 32768);
    *R1YCHAR = (*R1YCHAR | (32768UL << 16));
  }
  if ((dex[0] & 32) == 32) {
    *R1XCHAR = (*R1XCHAR | 16384);
    *R1YCHAR = (*R1YCHAR | (16384UL << 16));
  }
  if ((dex[0] & 16) == 16) {
    *R1XCHAR = (*R1XCHAR | 8192);
    *R1YCHAR = (*R1YCHAR | (8192 << 16));
  }
  if ((dex[0] & 8) == 8) {
    *R1XCHAR = (*R1XCHAR | 16384);
    *R1YCHAR = (*R1YCHAR | (16384 << 16));
  }
  if ((dex[0] & 4) == 4) {
    *R1XCHAR = (*R1XCHAR | 8192);
    *R1YCHAR = (*R1YCHAR | (8192 << 16));
  }
  if ((dex[0] & 2) == 2) {
    *R1XCHAR = (*R1XCHAR | 4096);
    *R1YCHAR = (*R1YCHAR | (4096 << 16));
  }
  if ((dex[0] & 1) == 1) {
    *R1XCHAR = (*R1XCHAR | 2048);
    *R1YCHAR = (*R1YCHAR | (2048 << 16));
  }

  if ((dex[1] & 128) == 128) {
    *R1XCHAR = (*R1XCHAR | 4096);
    *R1YCHAR = (*R1YCHAR | (4096 << 16));
  }
  if ((dex[1] & 64) == 64) {
    *R1XCHAR = (*R1XCHAR | 2048);
    *R1YCHAR = (*R1YCHAR | (2048 << 16));
  }
  if ((dex[1] & 32) == 32) {
    *R1XCHAR = (*R1XCHAR | 1024);
    *R1YCHAR = (*R1YCHAR | (1024 << 16));
  }
  if ((dex[1] & 16) == 16) {
    *R1XCHAR = (*R1XCHAR | 512);
    *R1YCHAR = (*R1YCHAR | (512 << 16));
  }
  if ((dex[1] & 8) == 8) {
    *R1XCHAR = (*R1XCHAR | 1024);
    *R1YCHAR = (*R1YCHAR | (1024 << 16));
  }
  if ((dex[1] & 4) == 4) {
    *R1XCHAR = (*R1XCHAR | 512);
    *R1YCHAR = (*R1YCHAR | (512 << 16));
  }
  if ((dex[1] & 2) == 2) {
    *R1XCHAR = (*R1XCHAR | 256);
    *R1YCHAR = (*R1YCHAR | (256 << 16));
  }
  if ((dex[1] & 1) == 1) {
    *R1XCHAR = (*R1XCHAR | 128);
    *R1YCHAR = (*R1YCHAR | (128 << 16));
  }

  if ((dex[2] & 128) == 128) {
    *R1XCHAR = (*R1XCHAR | 256);
    *R1YCHAR = (*R1YCHAR | (256 << 16));
  }
  if ((dex[2] & 64) == 64) {
    *R1XCHAR = (*R1XCHAR | 128);
    *R1YCHAR = (*R1YCHAR | (128 << 16));
  }
  if ((dex[2] & 32) == 32) {
    *R1XCHAR = (*R1XCHAR | 64);
    *R1YCHAR = (*R1YCHAR | (64 << 16));
  }
  if ((dex[2] & 16) == 16) {
    *R1XCHAR = (*R1XCHAR | 32);
    *R1YCHAR = (*R1YCHAR | (32 << 16));
  }
  if ((dex[2] & 8) == 8) {
    *R1XCHAR = (*R1XCHAR | 64);
    *R1YCHAR = (*R1YCHAR | (64 << 16));
  }
  if ((dex[2] & 4) == 4) {
    *R1XCHAR = (*R1XCHAR | 32);
    *R1YCHAR = (*R1YCHAR | (32 << 16));
  }
  if ((dex[2] & 2) == 2) {
    *R1XCHAR = (*R1XCHAR | 16);
    *R1YCHAR = (*R1YCHAR | (16 << 16));
  }
  if ((dex[2] & 1) == 1) {
    *R1XCHAR = (*R1XCHAR | 8);
    *R1YCHAR = (*R1YCHAR | (8 << 16));
  }

  if ((dex[3] & 128) == 128) {
    *R1XCHAR = (*R1XCHAR | 16);
    *R1YCHAR = (*R1YCHAR |(16 << 16));
  }
  if ((dex[3] & 64) == 64) {
    *R1XCHAR = (*R1XCHAR | 8);
    *R1YCHAR = (*R1YCHAR |(8 << 16));
  }
  if ((dex[3] & 32) == 32) {
    *R1XCHAR = (*R1XCHAR | 4);
    *R1YCHAR = (*R1YCHAR | (4 << 16));
  }
  if ((dex[3] & 16) == 16) {
    *R1XCHAR = (*R1XCHAR | 2);
    *R1YCHAR = (*R1YCHAR | (2 << 16));
  }
  if ((dex[3] & 8) == 8) {
    *R1XCHAR = (*R1XCHAR | 4);
    *R1YCHAR = (*R1YCHAR | (4 << 16));
  }
  if ((dex[3] & 4) == 4) {
    *R1XCHAR = (*R1XCHAR | 2);
    *R1YCHAR = (*R1YCHAR | (2 << 16));
  }
  if ((dex[3] & 2) == 2) {
    *R1XCHAR = (*R1XCHAR | 1);
    *R1YCHAR = (*R1YCHAR | (1 << 16));
  }
  if ((dex[3] & 1) == 1) {
    *R1XCHAR = (*R1XCHAR | 32768);
    *R1YCHAR = (*R1YCHAR | (32768UL << 16));
  }

  if (dey[0] & 8)
    *R1YCHAR = (*R1YCHAR | 16);
  if (dey[0] & 4)
    *R1YCHAR = (*R1YCHAR | 512);
  if (dey[0] & 2)
    *R1YCHAR = (*R1YCHAR | 32768);
  if (dey[0] & 1)
    *R1YCHAR = (*R1YCHAR | 128);

  if (dey[1] & 8)
    *R1YCHAR = (*R1YCHAR | 1);
  if (dey[1] & 4)
    *R1YCHAR = (*R1YCHAR | 256);
  if (dey[1] & 2)
    *R1YCHAR = (*R1YCHAR | 2048);
  if (dey[1] & 1)
    *R1YCHAR = (*R1YCHAR | 16384);

  if (dey[2] & 8)
    *R1YCHAR = (*R1YCHAR | 4);
  if (dey[2] & 4)
    *R1YCHAR = (*R1YCHAR | 1024);
  if (dey[2] & 2)
    *R1YCHAR = (*R1YCHAR | 2);
  if (dey[2] & 1)
    *R1YCHAR = (*R1YCHAR | 64);

  if (dey[3] & 8)
    *R1YCHAR = (*R1YCHAR | 8192);
  if (dey[3] & 4)
    *R1YCHAR = (*R1YCHAR | 32);
  if (dey[3] & 2)
    *R1YCHAR = (*R1YCHAR | 4096);
  if (dey[3] & 1)
    *R1YCHAR = (*R1YCHAR | 8);
}



// modified sort to cast locally first
void quicklySortLocal(UINT max) {
  UINT q, r, keycounter = 0, dmmy;

  // bubble sort entire PK2 array to group numbers together
  int* PK2LOCAL = (int*)&PK2[0];
  for (r = 0; r < max; r++) {
    for (q = 0; q < max; q++)
      if (PK2LOCAL[q + 1] < PK2LOCAL[q]) {
        dmmy = PK2LOCAL[q + 1];
        PK2LOCAL[q + 1] = PK2LOCAL[q];
        PK2LOCAL[q] = dmmy;
      }
  }

  for (r = 0; r < max; r++)
    if (PK2LOCAL[r + 1] != PK2LOCAL[r]) {
      keyArray[keycounter] = PK2LOCAL[r];
      keycounter++;
    }
}


/* MAIN FUNCTION */
int main(int argc, char *argv[]) {
  /******************* Variable Declaration and initializations *******************/
  COLUMN x, dx, dy, i, j;
  UINT input, curR1Y, curL, curV, cnt, origKey = 0, guessKey;
  UINT k, m, testKey;
  int r = 0;
  int ictr, local_copy_n;
  char xKey[8], gKey[8];

  ELEMENT DPS[256][256];
  ELEMENT DTS[256][16];
  UINT key32 = 0;
  UINT R1X = 0;
  UINT R1Y = 0;
  UINT C = 0;
  UINT C2 = 0;
  UINT keyCount[2][4096]; // keyCount[0][*] - actual key; keyCount[1][*] - count
  UINT R1XCHAR = 0;
  UINT R1YCHAR = 0;
  int dex[4], dey[4];
  double prob[4];
  UINT highCount;

  unsigned long long time1, time2, time6, time7;     // timing variables
  double ttime1, ttime2, ttime3, ttime4;             //

  upc_lock_t* resultlock;

  if (MYTHREAD == 0) {
    if (argc != 2) {
      printf("Run command syntax: <8-digit hex key>\n");
      fflush(stdout);
      upc_global_exit(-1);
    }

    key32_in = hex2UINT(argv[1]);
    cresultcnt = 0;
    sharedindex = 0;
    shared_n = 0;
  }
  upc_barrier;
  key32 = key32_in;

  for (r = 0; r < 256; r++)
    s[0][r] = s0[r];
  for (r = 0; r < 256; r++)
    s[1][r] = s1[r];
  for (r = 0; r < 256; r++)
    s[2][r] = s2[r];
  for (r = 0; r < 256; r++)
    s[3][r] = s3[r];

  /****************** Begin differential analysis of S-Boxes *********************
   * create difference pair and difference distribution tables
   */
  if (MYTHREAD == 0) {
    time1 = timer_now();
  }

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // NOTE: here each node does the same exact differential analysis
  // we could have one do it and send the results to all others,
  // but this saves a little on network traffic (and only takes about 100ms to do)
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (r = 0; r < 4; r++) {        // do each SBox one at a time:
    curL = 0;

    for (x = 0; x < 256; x++)         // create a difference pair table
      for (dx = 0; dx < 256; dx++)
        DPS[x][dx] = ((SIO(x, s[r])) ^ (SIO(x^dx, s[r])));



    for (dx = 0; dx < 256; dx++)      // create difference distribution table
      for (dy = 0; dy < 16; dy++) {
        cnt = 0;
        for (i = 0; i < 256; i++)
          if ((DPS[i][dx]) == dy)
            cnt++;

        DTS[dx][dy] = cnt;
      }



    for (i = 0; i < 256; i++)          // find most probable input/ouput difference pair
      for (j = 0; j < 16; j++) {
        curV = DTS[i][j];
        if ((curV > curL) && (curV != 256)) {
          curL = curV;
          dex[r] = i;
          dey[r] = j;
        }
      }
    prob[r] = ((double)curL) / 256;

  }                             // end analysis of this SBox

  extendDC(&R1XCHAR, &R1YCHAR, dex, dey);       // after all SBoxes analyzes, find optimum
  // round characteristic (only done once)

  if (MYTHREAD == 0) {
    time2 = timer_now();
  }

  /*********************** END ANALYSIS OF SBOXES ********************************
   */

  /* From here the chosen plaintext and ciphertext files are analyzed;
   * this portion of the algorithm takes the majority of the time
   */

  // grab a lock
  resultlock = upc_all_lock_alloc();

  upc_forall(input = 0; input < NUMPAIRS; input++; input) {
    // grab all crypts that match up
    docrypt(key32, input, 0, &R1X, &R1Y, &C, &C2);                    // perform 2 encryptions
    curR1Y = R1Y;                      // per iteration of loop
    docrypt(key32, (input ^ R1XCHAR), 1, &R1X, &R1Y, &C, &C2);
    if ((R1Y ^ curR1Y) == (R1YCHAR)) {
      // lock the result array & stick it in there
      upc_lock(resultlock);
      cresult[cresultcnt].r1y = R1Y;
      cresult[cresultcnt].curr1y = curR1Y;
      cresult[cresultcnt].c = C;
      cresult[cresultcnt].c2 = C2;
      cresult[cresultcnt].indexfound = input;
      cresultcnt++;
      upc_unlock(resultlock);
      if (cresultcnt == MAXCRESULT) {
        printf("Error: too many results.\n");
        fflush(stdout);
        upc_global_exit(-1);
      }
    }
  }
  upc_barrier;

  // To ensure same traversal as sequential version, we have to make sure
  // we go over the first two results during the MAINKEYLOOP loop
  // (use a simple selection sort)
  if (MYTHREAD == 0) {
    unsigned long long a;
    struct cryptresult tmp;
    a = timer_now();
    for (i = 0; i < cresultcnt; i++) {
      for (j = i + 1; j < cresultcnt; j++) {
        if (cresult[j].indexfound < cresult[i].indexfound) {
          // swap ith and jth elements
          tmp = cresult[j];
          cresult[j] = cresult[i];
          cresult[i] = tmp;
        }
      }
    }
    printf("First sort took %0.9f\n", timer_elapsed(a));
    fflush(stdout);
  }
  upc_barrier;

  // loop over what we found before...
  for (ictr = 0; ictr < cresultcnt; ictr++) {
    struct cryptresult cr = cresult[ictr];
    R1Y = cr.r1y;
    curR1Y = cr.curr1y;
    C = cr.c;
    C2 = cr.c2;

    if (ictr < 2) {
      upc_forall(m = 0; m < MAINKEYLOOP; m++; m) {
        unsigned long long a = timer_now();
        upc_forall(k = 0; k < 1048576; k++; continue) {
          testKey = (1048576 * m) + k;
          if ((lastRound(curR1Y, testKey) == C) && (lastRound(R1Y, testKey) == C2)) {
            printf("Got %u from %u, %u\n", testKey, C, C2);
            fflush(stdout);
            upc_lock(resultlock);
            PK2[sharedindex] = testKey;
            sharedindex++;
            upc_unlock(resultlock);
          }
        }
        printf("iteration %d took %0.9f\n", m, timer_elapsed(a));
        fflush(stdout);
      }

      if (ictr == 1) {
        upc_barrier;
        if (MYTHREAD == 0) {
          unsigned long long a = timer_now();
          // only have the master do the sort, since it won't take that long
          // do the sort (the PK2 array has affinity to the master thread)
          quicklySortLocal(sharedindex);
          printf("sort took %0.9f\n", timer_elapsed(a));
          fflush(stdout);
          shared_n = sharedindex;
        }
        upc_fence;
        upc_barrier;
        local_copy_n = shared_n;
      }
    } // end if ictr < 2
    else {
      // loop through, making sure to group by who has affinity to keyArray's element
      // that the iteration is looking at
      upc_forall(m = 0; m < local_copy_n; m++; &keyArray[m]) {
        if ((lastRound(curR1Y, keyArray[m]) == C) && (lastRound(R1Y, keyArray[m]) == C2)) {
          upc_lock(resultlock);
          PK2[sharedindex] = keyArray[m];
          sharedindex++;
          upc_unlock(resultlock);
        }
      }
    } // end if ictr >= 3
  } // end loop through crypt results

  upc_barrier;

  if (MYTHREAD == 0) {
    time6 = timer_now();

    /*******************   END PLAINTEXT / CIPHERTEXT ANALYSIS ******************
     */

    if (sharedindex == 0) {
      printf("\n\nNo matches found; analysis unsuccessful...\n");
      fflush(stdout);
      return -1;
    } else {

      /* The sortNcount function sorts the PK2 array from lowest to highest,
       * and counts the number of occurences of each number, reporting the
       * highest: determines the most probable subkey for round 2
       */
      guessKey = sortNcount(sharedindex, keyCount, &highCount);
      time7 = timer_now();

      /***************************** END MAIN PROGRAM ******************************
       * - all that remains is timing and displaying results
       */

      for (k = 0; k < SUBKEYSIZE; k++)
        if ((1 << (KEYSIZE - pInverse[k] - 1)) & guessKey)
          origKey |= 1 << (KEYSIZE - k - 1);

      UINT2hex(origKey, xKey);
      UINT2hex(guessKey, gKey);

      printf("\n\nGuessed subkey (round 2):  ");
      for (k = 0; k < 8; k++)
        printf("%c", gKey[k]);
      printf("\nCorresponding Key:      ");
      for (k = 0; k < 8; k++)
        printf("%c", xKey[k]);


      ttime1 = timer_elapsed2(time1, time2);
      ttime2 = timer_elapsed2(time2, time6);
      ttime3 = timer_elapsed2(time6, time7);
      ttime4 = timer_elapsed2(time1, time7);

      printf("\n\nUser specified key:  %s", argv[1]);
      printf("\n\nTotal number of occurences:  %u\n", highCount);
      printf("Elements in PK2[]:    %u\n\n\n", sharedindex);
      printf("Execution Time Info:\n\n");
      printf("Time to analyze SBoxes:      %.9lf s\n", ttime1);
      printf("Time to analyze PT/CT pairs:    %.9lf s\n", ttime2);
      printf("Time to search data for best key:  %.9lf s\n\n", ttime3);
      printf("Total Execution Time:    %.9lf s\n\n", ttime4);
      fflush(stdout);
    }
  } // end if MYTHREAD==0

  return 0;
}
