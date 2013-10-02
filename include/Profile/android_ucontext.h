/*
 * Android doesn't provide <ucontext.h> so we add our own here
 */
#ifdef TAU_ANDROID

#ifndef _ANDROID_UCONTEXT_H_
#define _ANDROID_UCONTEXT_H_

#include <asm/signal.h>     /* sigset_t */
#include <asm/sigcontext.h> /* struct sigcontext */

struct ucontext {
    unsigned long  uc_flags;
    struct ucontext  *uc_link;
    stack_t  uc_stack;
    struct sigcontext uc_mcontext;
    sigset_t  uc_sigmask;
    /* Allow for uc_sigmask growth.  Glibc uses a 1024-bit sigset_t.  */
    int  _unused[32 - (sizeof (sigset_t) / sizeof (int))];
    /* Last for extensibility.  Eight byte aligned because some
       coprocessors require eight byte alignment.  */
    unsigned long  uc_regspace[128] __attribute__((__aligned__(8)));
};

typedef struct ucontext ucontext_t;

#endif /* _ANDROID_UCONTEXT_H_ */

#endif /* TAU_ANDROID */
