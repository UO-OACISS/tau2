!****************************************************************************
!*			TAU Portable Profiling Package			   **
!*			http://www.cs.uoregon.edu/research/tau	           **
!****************************************************************************
!*    Copyright 1997-2009					   	   **
!*    Department of Computer and Information Science, University of Oregon **
!*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
!****************************************************************************
!****************************************************************************
!*	File 		: TauFAPI.h					   **
!*	Description 	: TAU Profiling Package API for Fortran		   **
!*	Contact		: tau-bugs@cs.uoregon.edu                	   **
!*	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
!****************************************************************************

        integer TAU_DEFAULT
	integer TAU_MESSAGE
	integer TAU_PETE
	integer TAU_VIZ
	integer TAU_ASSIGN
	integer TAU_IO
	integer TAU_FIELD
	integer TAU_LAYOUT
	integer TAU_SPARSE
	integer TAU_DOMAINMAP
	integer TAU_UTILITY
	integer TAU_REGION
	integer TAU_PARTICLE
	integer TAU_MESHES
	integer TAU_SUBFIELD
	integer TAU_COMMUNICATION
	integer TAU_DESCRIPTOR_OVERHEAD
	integer TAU_BLITZ
	integer TAU_HPCXX
	integer TAU_FFT
	integer TAU_ACLMPL
	integer TAU_PAWS1
	integer TAU_PAWS2
	integer TAU_PAWS3
	integer TAU_USER4
	integer TAU_USER3
	integer TAU_USER2
	integer TAU_USER1
	integer TAU_USER

	parameter (TAU_DEFAULT 			= Z"ffffffff")	
	parameter (TAU_MESSAGE 			= Z"00000001")	
	parameter (TAU_PETE 			= Z"00000002")	
	parameter (TAU_VIZ 			= Z"00000004")	
	parameter (TAU_ASSIGN 			= Z"00000008")	
	parameter (TAU_IO 			= Z"00000010")	
	parameter (TAU_FIELD 			= Z"00000020")	
	parameter (TAU_LAYOUT 			= Z"00000040")	
	parameter (TAU_SPARSE 			= Z"00000080")	
	parameter (TAU_DOMAINMAP 		= Z"00000100")	
	parameter (TAU_UTILITY 			= Z"00000200")	
	parameter (TAU_REGION 			= Z"00000400")	
	parameter (TAU_PARTICLE 		= Z"00000800")	
	parameter (TAU_MESHES 			= Z"00001000")	
	parameter (TAU_SUBFIELD 		= Z"00002000")	
	parameter (TAU_COMMUNICATION 		= Z"00004000")	
	parameter (TAU_DESCRIPTOR_OVERHEAD 	= Z"00008000")	
	parameter (TAU_BLITZ			= Z"00010000")	
	parameter (TAU_HPCXX 			= Z"00020000")	
	parameter (TAU_FFT 			= Z"00100000")	
	parameter (TAU_ACLMPL 			= Z"00200000")	
	parameter (TAU_PAWS1 			= Z"00400000")	
	parameter (TAU_PAWS2 			= Z"00800000")	
	parameter (TAU_PAWS3 			= Z"01000000")	
	parameter (TAU_USER4 			= Z"08000000")	
	parameter (TAU_USER3 			= Z"10000000")	
	parameter (TAU_USER2 			= Z"20000000")	
	parameter (TAU_USER1 			= Z"40000000")	
	parameter (TAU_USER 			= Z"80000000")	
