
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
#else /* IA64 */
inline unsigned long long getLinuxHighResolutionTscCounter(void)
{
   unsigned long high, low;
   __asm__ __volatile__(".byte 0x0f,0x31" : "=a" (low), "=d" (high));
   return ((unsigned long long) high << 32) + low;
}
#endif /* IA64 */

  
