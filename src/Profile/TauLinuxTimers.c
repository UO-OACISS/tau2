
///////////////////////////////////////////////////////////////////////////
// High resolution timers. Compile with gcc under Linux on IA-32 systems. 
///////////////////////////////////////////////////////////////////////////
inline unsigned long long getLinuxHighResolutionTscCounter(void)
{
   unsigned long high, low;
   __asm__ __volatile__(".byte 0x0f,0x31" : "=a" (low), "=d" (high));
   return ((unsigned long long) high << 32) + low;
}

