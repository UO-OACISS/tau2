//#include "stdafx.h"
#include <stdio.h>
#include <string.h>

int  optind  = 1;
char *optarg  = NULL;
int  opterr  = 1;

int getopt (int argc, char *argv[], char *optstring)
{
 char *cptr;
 static char scan_char   = 0;
 static int letter_index = 0;

 optarg = NULL;

 /*
  *  check if at the beginning of a group of option letters
  */
 if (letter_index == 0)
 {
  if (optind >= argc)
   return -1;
  /*
   *  skip to next arg unless this is first
   */
  if (optind != 1)
   optind++;
 }

 while ((optind < argc) || (letter_index != 0))
 {
  optarg = argv[optind];

  /*
   *  check if at the beginning of a group of option letters
   */
  if (letter_index == 0)
  {
   if (optind >= argc)
   {
    optarg = NULL;
    return -1;
   }
   if ((argv[optind][0] == '/') || (argv[optind][0] == '-'))
   {
    /*
     *  check if "//"
     */
    if (argv[optind][1] == argv[optind][0])
    {
     return -1;
    }
    /*
     *  only one / -- just skip it
     */
    letter_index++;
   }
   else
    return -1;
  }

  scan_char = optarg[letter_index++];

  /*
   *  check if end of option letter group
   */
  if (scan_char == 0)
  {
   letter_index = 0;
   optind++;
   optarg = NULL;

   continue;
  }

  /*
   *  check if argument is a "dbug" command
   */
  if ((optarg[0] == '/') || (optarg[0] == '-'))
  {
   if (optarg[1] == '#')
   {
    letter_index = 0;
    optind++;
    optarg = NULL;

    continue;
   }
   else
    break;
  }
 }

 if (scan_char == 0)
  return -1;

 if (optind >= argc)
 {
  optarg = NULL;
  return -1;
 }

 cptr = strchr(optstring, scan_char);  
 /*
  *  check if it's a valid command letter
  */
 if (cptr == NULL || scan_char == ':')
  return ('?');

 cptr++;

 /*
  *  check if this command takes an argument
  */
 if (*cptr == ':')
 {
  if (optind < argc)
   optarg = argv[++optind];
  else
   optarg = NULL;

  letter_index = 0;
 }
 else
  optarg = NULL;

 return scan_char;
}



