/*
 * High resolution timers. Compile with gcc under Linux on IA-32/IA-64 systems. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus 
extern "C" { 
#endif /* __cplusplus */

#ifdef __ia64__
unsigned long long getLinuxHighResolutionTscCounter(void) {
  unsigned long long tmp;
  __asm__ __volatile__("mov %0=ar.itc" : "=r"(tmp) :: "memory");
  return tmp;
}
#elif __aarch64__
unsigned long long getLinuxHighResolutionTscCounter(void) {
  unsigned long long tmp;
  __asm__ __volatile__("mrs %0, CNTVCT_EL0" : "=r" (tmp));
  return tmp;
}
#elif __powerpc__
unsigned long long getLinuxHighResolutionTscCounter(void) {
  unsigned long long tmp;
  unsigned int Low, HighB, HighA;

  do {
    asm volatile ("mftbu %0" : "=r"(HighB));
    asm volatile ("mftb %0" : "=r"(Low));
    asm volatile ("mftbu %0" : "=r"(HighA));
  } while (HighB != HighA);
  tmp = ((unsigned long long)HighA<<32) | ((unsigned long long)Low);
  return tmp;
}
#else
unsigned long long getLinuxHighResolutionTscCounter(void) {
   unsigned long high, low;
   __asm__ __volatile__(".byte 0x0f,0x31" : "=a" (low), "=d" (high));
   return ((unsigned long long) high << 32) + low;
}
#endif /* IA64 */

  
int TauReadFullLine(char *line, FILE *fp) {
  int ch, i;
  i = 0; 
  while ( (ch = fgetc(fp)) && ch != EOF && ch != (int) '\n') {
    line[i++] = (unsigned char) ch;
  }
  line[i] = '\0'; 
  if (ch == EOF) {
    return -1;
  }
  return i; 
}

double TauGetMHzRatings(void) {
  float ret = 0;
  char line[4096];
  double rating = 0;
  char *apple_cmd = "sysctl hw.cpufrequency | sed 's/^.*: //'";
  FILE *fp = fopen("/proc/cpuinfo", "r");

#ifdef __aarch64__ 

  long long freq; 
  __asm__ __volatile__("mrs %0, CNTFRQ_EL0" : "=r" (freq));
  rating = (double) freq/1.0e6;
  return rating; 

#endif /* __aarch64__ */

  if (fp) {
    while (TauReadFullLine(line, fp) != -1) {
      if (strncmp(line, "cpu MHz", 7) == 0) {
        sscanf(line,"cpu MHz         : %f", &ret);
        return (double) ret; 
      }
      if (strncmp(line, "timebase", 8) == 0) {
        sscanf(line,"timebase        : %f", &ret);
        return (double) ret / 1.0e6; 
      }
    }
  } else {
    /* assume apple */
    fp = popen(apple_cmd,"r");
    if (fp == NULL) {
      perror("/proc/cpuinfo file not found:");
    } else {
      while (fgets(line, 4096, fp) != NULL) {
	rating = atof(line);
      }
    } 
    pclose(fp);
    return rating/1E6; /* Apple returns Hz not MHz. Convert to MHz */
  }
  return (double) ret;
}
  
double TauGetMHz(void) {
  static double init = 0;
  static double ratings;
  if (init == 0) {
    ratings = TauGetMHzRatings();
    init = 1;
  }
  return ratings;
}

#ifdef __cplusplus {
}
#endif /* __cplusplus */
