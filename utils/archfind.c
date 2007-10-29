/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1993,1995             */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

# ifdef AMIX /* Amiga UNIX */
#  define _havehosttype_
ARCHamiga
# endif /* AMIX */

# ifdef __PARAGON__
#  define _havehosttype_
ARCHparagon
# endif 

# ifdef _VMS_POSIX
#  define _havehosttype_
ARCHvms_posix
# endif /* _VMS_POSIX */

# if (defined(vax) || defined(__vax)) && !defined(_havehosttype_)
#  define _havehosttype_
ARCHvax
# endif /* vax || __vax && !_havehosttype_ */

# ifdef hp9000 /* hp9000 running MORE/bsd */
#  ifdef hp300
#   define _havehosttype_
ARCHhp300
#  endif 
#  ifdef hp800
#   define _havehosttype_
ARCHhp800
#  endif 
#  ifndef _havehosttype_
#   define _havehosttype_
ARCHhp9000
#  endif 
# endif /* hp9000 */

# if defined(sun) || defined(__sun__)
#  if defined(mc68010) || defined(__mc68010__)
#   define _havehosttype_
ARCHsun2
#  endif /* mc68010 */
#  if defined(mc68020) || defined(__mc68010__)
#   define _havehosttype_
ARCHsun3
#  endif /* mc68020 */

#  if defined(__svr4__) && (defined(sparc) && defined(__sparc__))
#   define _havehosttype_
ARCHsolaris2
#  endif /* solaris */

#  if (defined(sparc) || defined(__sparc__)) && !defined(__svr4__)
#   define _havehosttype_
ARCHsun4
#  endif /* sparc */

#  if defined(i386) || defined(__i386__)
#   define _havehosttype_
ARCHsun386i
#  endif /* i386 */
#  ifndef _havehosttype_
#   define _havehosttype_
ARCHsun
#  endif 
# endif /* sun */

# ifdef pyr /* pyramid */
#  define _havehosttype_
ARCHpyramid
# endif /* pyr */

# ifdef tahoe /* tahoe */
#  define _havehosttype_
ARCHtahoe
# endif /* tahoe */

# ifdef ibm032 /* from Jak Kirman */
#  define _havehosttype_
ARCHrt
# endif /* ibm032 */

# ifdef aiws /* not to be confused with the above */
#  define _havehosttype_
ARCHrtpc
# endif /* aiws */

# ifdef _AIX370
#  define _havehosttype_
ARCHaix370
# endif /* _AIX370 */

# ifdef _IBMESA
#  define _havehosttype_
ARCHaixESA
# endif /* _IBMESA */

# if (defined( _IBMR2) || defined( _AIX ))
#  define _havehosttype_
ARCHrs6000
# endif /* _IBMR2 */

# ifdef _AIXPS2 /* AIX on a PS/2 */
#  define _havehosttype_
ARCHps2
# endif /* _AIXPS2 */

# ifdef OREO
#  define _havehosttype_
ARCHmac2
# endif /* OREO */

# ifdef __hpux
#  if defined(__hp9000s700) && !defined(_havehosttype_)
#   define _havehosttype_
ARCHhp9000s700
#  endif /* __hp9000s700 */
#  if (defined(__hp9000s800) || defined(hp9000s800)) && !defined(_havehosttype_)
#   define _havehosttype_
ARCHhp9000s800
#  endif /* __hp9000s800 || hp9000s800 */
#  if (defined(__hp9000s300) || defined(hp9000s300)) && !defined(_havehosttype_)
#   define _havehosttype_
ARCHhp9000s300
#  endif /* __hp9000s800 || hp9000s300 */
# if defined(hp9000s500) && !defined(_havehosttype_)
#  define _havehosttype_
ARCHhp9000s500
# endif /* hp9000s500 */
#  ifndef _havehosttype_
#   define _havehosttype_
ARCHhp
#  endif /* _havehosttype_ */
# endif /* __hpux */

# ifdef apollo
#  define _havehosttype_
ARCHapollo
# endif 

# ifdef u3b20d
#  define _havehosttype_
ARCHatt3b20
# endif /* u3b20d */

# ifdef u3b15
#  define _havehosttype_
ARCHatt3b15
# endif /* u3b15 */

# ifdef u3b5
#  define _havehosttype_
ARCHatt3b5
# endif /* u3b5 */

# ifdef u3b2
#  define _havehosttype_
ARCHatt3b2
# endif /* u3b2 */

#ifdef _MINIX
# define _havehosttype_
# ifdef i386
ARCHminix386
# else /* minix ? amoeba or mac? */
ARCHminix
# endif /* i386 */
#endif /* _MINIX */

#if defined(i386) && defined(linux)
# define _havehosttype_
ARCHi386_linux
#endif

#if defined(__x86_64) && defined(linux)
# define _havehosttype_
ARCHx86_64
#endif

#if defined(i386) && defined(__EMX__)
# define _havehosttype_
ARCHi386_emx
#endif /* i386 && __EMX__ */

# ifdef __386BSD__
# define _havehosttype_
ARCH386BSD
# endif /* __386BSD__ */

# if defined(i386) && defined(bsdi)
#  define _havehosttype_
ARCHbsd386
# endif /* i386 && bsdi */

# ifdef COHERENT
#  define _havehosttype_
#  ifdef _I386
ARCHcoh386
#  else
ARCHcoherent
#  endif /* _I386 */
# endif /* COHERENT */

# if defined(i386) && SYSVREL > 0

#  if !defined(_havehosttype_) && (defined(ISC) || defined(ISC202))
#   define _havehosttype_
ARCHisc386
#  endif /* !_havehosttype_ && (ISC || ISC202) */

#  if !defined(_havehosttype_) && defined(SCO)
#   define _havehosttype_
ARCHsco386
#  endif /* !_havehosttype_ && SCO */

#  if !defined(_havehosttype_) && defined(INTEL)
#   define _havehosttype_
ARCHintel386
#  endif /* !_havehosttype_ && INTEL */

#  ifndef _havehosttype_
#   define _havehosttype_
ARCHi386
#  endif /* _havehosttype_ */

# endif 

#ifdef UNIXPC
# define _havehosttype_
ARCHunixpc
#endif /* UNIXPC/att3b1/att7300 */

#ifdef alliant
# ifdef mc68000
#  define _havehosttype_
ARCHalliant_fx80
# endif /* mc68000 */
# ifdef i860 
#  define _havehosttype_
ARCHalliant_fx2800
# endif /* i860 */
# ifndef _havehosttype_
#  define _havehosttype_
ARCHalliant
# endif /* _havehosttype_ */
#endif  /* alliant */

# if defined(i386) && defined(MACH)
#  define _havehosttype_
ARCHi386_mach
# endif 

# if defined(sequent) || defined(_SEQUENT_)
#  define _havehosttype_
#  ifdef i386
#   ifdef sequent
ARCHsymmetry
#    ifndef LOCALSTR
#     define LOCALSTR	" (Dynix/3)"
#    endif /* LOCALSTR */
#   else
ARCHptx
#    ifndef LOCALSTR
#     define LOCALSTR	" (Dynix/ptx)"
#    endif /* LOCALSTR */
#   endif 
#  else
ARCHbalance
#   ifndef LOCALSTR
#    define LOCALSTR	" (Dynix/3)"
#   endif /* LOCALSTR */
#  endif 
# else /* !sequent */
#  ifdef ns32000
#   define _havehosttype_
#   ifdef CMUCS			/* hack for Mach (in the true spirit of CMU) */
ARCHmultimax
#   else /* CMUCS */
ARCHns32000
#   endif /* CMUCS */
#  endif /* ns32000 */
# endif /* sequent */

# ifdef convex
#  define _havehosttype_
ARCHconvex
# endif /* convex */

# ifdef butterfly
#  define _havehosttype_
#  if defined(BFLY2) || defined(__BFLY2__)
ARCHtc2000
#  else
ARCHgp1000
#  endif
# endif /* butterfly */

# ifdef NeXT
#  define _havehosttype_
ARCHnext
# endif /* NeXT */

/* From Kazuhiro Honda <honda@mt.cs.keio.ac.jp> */
# ifdef sony_news
#  define _havehosttype_
#  ifdef mips /* Sony NEWS based on a r3000 */
ARCHnews_mips
#  else
ARCHnews
#  endif 
# endif /* sony_news */

# if defined(mips) || defined(__mips)
#  define _havehosttype_
#  if defined(MIPSEL) || defined(__MIPSEL)
#   if defined(ultrix) || defined(__ultrix)
ARCHdecstation
#   else
ARCHmips
#   endif /* ultrix || __ultrix */
#  endif /* MIPSEL || __MIPSEL */
#  if defined(MIPSEB) || defined(__MIPSEB)
#   if defined(ultrix) || defined(__ultrix)
ARCHdecmips
#   else
#    ifdef sgi /* sgi */
/* old code #     if ((__mips == 1) || (__mips == 2)) */
#if (_MIPS_ISA == _MIPS_ISA_MIPS1 || _MIPS_ISA == _MIPS_ISA_MIPS2)
ARCHsgi4k
#     else
ARCHsgi8k
#     endif /* __mips */
#    else
#     ifdef sony_news
ARCHnews_mips
#     else
ARCHmips
#     endif /* sony_news */
#    endif /* sgi */
#   endif /* ultrix || __ultrix */
#  endif /* MIPSEB || __MIPSEB */
# endif /* mips || __mips */

#if defined(__alpha)
#  define _havehosttype_
ARCHalpha
#endif

#if defined(__APPLE__)
#  define _havehosttype_ 
ARCHapple 
# endif 

#if defined(__hiuxmpp)
#  define _havehosttype_ 
ARCHhitachi
# endif 

#if defined(_SX)
#  define _havehosttype_ 
ARCHnec
# endif 

# if defined(m88k) || defined(__m88k__)
#   ifndef _havehosttype_
#     define _havehosttype_
ARCHm88k
#   endif
# endif 

# ifdef masscomp			/* Added, DAS DEC-90. */
#  define _havehosttype_
ARCHmasscomp
# endif /* masscomp */

# ifdef GOULD_NP1
#  define _havehosttype_
ARCHgould_np1
# endif /* GOULD_NP1 */

# ifdef SXA
#  define _havehosttype_
ARCHpfa50
#  ifdef  _BSDX_
#   ifndef LOCALSTR
#    define LOCALSTR	" (SX/A E60+BSDX)"
#   endif /* LOCALSTR */
#  else
#   ifndef LOCALSTR
#    define LOCALSTR	" (SX/A E60)"
#   endif /* LOCALSTR */
#  endif 
# endif /* PFU/Fujitsu A-xx computer */

# ifdef titan
#  define _havehosttype_
    /* Ken Laprade <laprade@trantor.harris-atd.com> */
ARCHtitan
# endif /* titan */

# ifdef stellar
#  define _havehosttype_
ARCHstellar
# endif /* stellar */

# ifdef sgi
/* Iris 4D is in the mips section; these are the 68k machines. */
#  ifdef m68000
#   define _havehosttype_
    /* Vince Del Vecchio <vd09@andrew.cmu.edu> */
ARCHiris3d
#  endif
# endif /* sgi */

# ifdef uts
#  define _havehosttype_
ARCHamdahl
# endif /* uts */

# ifdef UTek
#  define _havehosttype_
ARCHtek4300
# endif /* UTek */

# ifdef UTekV
#  define _havehosttype_
ARCHtekXD88
# endif /* UTekV */

# ifdef OPUS
#  define _havehosttype_
ARCHopus
# endif /* OPUS */

# ifdef eta10
#  define _havehosttype_
   /* Bruce Woodcock <woodcock@mentor.cc.purdue.edu> */
ARCHeta10
# endif /* eta10 */

# ifdef cray                  /* CRAYTEST */
#   include "archcray.h"
# endif /* cray */            /* CRAYTEST */

# ifdef NDIX
#  define _havehosttype_
   /* B|rje Josefsson <bj@dc.luth.se> */
ARCHnd500
# endif /* NDIX */

# if defined(sysV68)
#  define _havehosttype_
ARCHsysV68
# endif /* sysV68 */

# if defined(sysV88)
#  define _havehosttype_
ARCHsysV88
# endif /* sysV88 */

# if defined(i860) && !defined(_havehosttype_)
#  define _havehosttype_
   /* Tasos Kotsikonas <tasos@avs.com> */
ARCHvistra800 /* Stardent Vistra */
# endif /* i860  && !_havehosttype_ */

# ifndef _havehosttype_
#  if defined(mc68000) || defined(__mc68000__) || defined(mc68k32)
#   define _havehosttype_
ARCHm68k
#  endif 
# endif

# ifndef _havehosttype
#   if defined (__ksr__)
#    define _havehosttype_
ARCHksr1
#   endif
# endif


# ifndef _havehosttype
#   if defined (__CYGWIN__)
#    define _havehosttype_
ARCHcygwin
#   endif
# endif


# ifndef _havehosttype_
#  define _havehosttype_
    /* Default to something reasonable */
ARCHunknown
# endif 
# undef _havehosttype_
