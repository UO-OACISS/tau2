
///////////////////////////////////////////////////////////////////////////
// High resolution timers. Compile with gcc under Linux on IA-32/IA-64 systems. 
///////////////////////////////////////////////////////////////////////////


#ifdef __ia64__
inline unsigned long long getLinuxHighResolutionTscCounter(void)
{
  unsigned long long tmp;
  __asm__ __volatile__("mov %0=ar.itc" : "=r"(tmp) :: "memory");
  return tmp;
}
#elif __powerpc__
inline unsigned long long getLinuxHighResolutionTscCounter(void)
{
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
inline unsigned long long getLinuxHighResolutionTscCounter(void)
{
   unsigned long high, low;
   __asm__ __volatile__(".byte 0x0f,0x31" : "=a" (low), "=d" (high));
   return ((unsigned long long) high << 32) + low;
}
#endif /* IA64 */

  
