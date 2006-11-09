/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.acl.lanl.gov/tau                        **
*****************************************************************************
**    Copyright 1999                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : ktau_proc_interface.c                           **
**      Description     : Kernel-space  /proc interface                   **
**      Author          : Aroon Nataraj                                   **
**                      : Surave Suthikulpanit                            **
**      Contact         : anataraj@cs.uoregon.edu                         **
**                      : suravee@cs.uoregon.edu                          **
**      Flags           :                                                 **
**      Documentation   :                                                 **
***************************************************************************/


//kernel/ktau/ktau_proc_interface.c
//

//Needed for compatibility between kern-space &
//user-space. But including linux/types.h causes
//a lot of conflicts. Hence HACK: typedefing
//seperately.
//ARCH-DEPEND
typedef int pid_t;

/* ktau KERNEL headers */
#include <linux/ktau/ktau_proc_data.h>

/* ktau user-spc-interface headers */
//#include <ktau_proc_interface.h>
#include "../../include/Profile/ktau_proc_interface.h"

/* user-sp headers */
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/* User-Space Application Interface Functions */


/* Ktau merge & gettime overloaded syscall */
/*
#include <asm/unistd.h>
#include <sys/time.h>
//declare the sys_ktau_gettimeofday syscall
#define __NR_ktau_gettimeofday 271 //super-hack - 271 is only for neuronic
_syscall2(int,ktau_gettimeofday,struct timeval *,tv,struct timezone *,tz);
//_syscall1(int,ktau_gettimeofday,struct timeval *,tv);
*/

/* Read & Purge PIDs 				    *
 * If nopids = 1, then used for single pid          *
 * If nopids > 1, then used for multiple pids       *
 * if nopids = 0 & pid=null, then used for ALL Pids *
 */

#define MAX_KTAU_PROC_PATH 1024

#define PROFILE_PATH "/proc/ktau/profile"
#define TRACE_PATH "/proc/ktau/trace"

/* read_size: Used to query /proc/ktau for SIZE of profile(s). *
 ***************************************************************
 * Arguments:
 * type:        KTAU_PROFILE or KTAU_TRACE
 *
 * self:        if reading self -> set to 1, otherwise set to zero
 *
 * pid:         pointer to a single pid , or address of array of pids
 *              Can be NULL (if nopids is ZERO) --> size of FULL sys. prof
 *
 * nopids:      no of pids being pointed to above
 *              Can be ZERO --> request size of entire system prof
 *
 * tol:         tolerance paramter. to ignore, set to ZERO
 *
 * flags:       <TO BE DONE> . ignore, set to NULL.
 *
 * compratio:	This is a fraction between 0 & 1 (can be zero or 1 also)
 * 		-1 --> means USE DEFAULT VALUE
 * 		Compensation-Ratio Used to scale-up the size returned 
 * 		from /proc, so that when a call is made to read_data, 
 * 		the size is larger to accomodate any changes to profile. 
 * 		NOTE: To Ignore SET to -1. 
 *
 * Returns:
 * On Success:  total size of profile(s) of pid(s)
 * On Error:    -1; Other negative error codes reserved.
 *
 * Constraints:
 * None.
 */
long read_size(int type, int self, pid_t *pid, unsigned int nopids, unsigned long tol, unsigned int* flags, float compratio)
{
	int ret_value = 0;
	int fd = -1;
	char path[MAX_KTAU_PROC_PATH+1];
	ktau_package_t ktau_pkg;
	ktau_size_t size;
	unsigned long lc_size = 0;
	unsigned int cmd = 0;
	char* hdrbuf = NULL;


	//printf("DEBUG: size of o_ent %d\n",sizeof(o_ent));
	//printf("DEBUG: size of h_ent %d\n",sizeof(h_ent));
	//printf("DEBUG: size of ktau_data %d\n",sizeof(ktau_data));
	//printf("DEBUG: size of ktau_timer %d\n",sizeof(ktau_timer));
	//printf("DEBUG: size of ktau_pcounter %d\n",sizeof(ktau_perf_counter));


	/* initialize ktau_size_t with user-args */
	size.tol.tol_in = tol;
	size.flags = 0;
	if(flags)
		size.flags = *flags;

	switch(type) {
	case KTAU_PROFILE:
		snprintf(path, MAX_KTAU_PROC_PATH, PROFILE_PATH);
		break;
	case KTAU_TRACE:
		snprintf(path, MAX_KTAU_PROC_PATH, TRACE_PATH);
		break;
	default:
		//def is profile
		snprintf(path, MAX_KTAU_PROC_PATH, PROFILE_PATH);
	}


	if((nopids == 1) && (self == 1)) /* single pid get size : use the read method */
	{
		switch(type) {
		case KTAU_PROFILE:
			cmd = CMD_SIZE;
			break;
		case KTAU_TRACE:
			cmd = CMD_TRACE_SIZE;
			break;
		default:
			//def is profile
			cmd = CMD_SIZE;
		}

		//fd = open(path,O_RDONLY);	
		fd = open(path,O_RDONLY);	
		if(fd < 0)
		{
			perror("open:");
			return fd;
		}

		ktau_pkg.op.size = size;
		ktau_pkg.type = cmd;
		ktau_pkg.pid = *pid;
		ktau_pkg.nopids = nopids;
		ret_value = 0;
		
		/* CHANGED BELOW: Using ONLY ioctl now. And can only do SELF OR ALL
		 * do {
			ret_value = write(fd, &ktau_pkg, sizeof(ktau_package_t));
		} while((ret_value < 0) && (errno == EINTR));
		*/
		ret_value = ioctl(fd, cmd, &ktau_pkg);

		lc_size = ktau_pkg.op.size.size;
		if(flags)
			*flags = ktau_pkg.op.size.flags;

	}
	else //if (nopids == 0) /* ALL Pids: use IOCTL method */
	{
		switch(type) {
		case KTAU_PROFILE:
			cmd = CMD_ALL_SIZE;
			break;
		case KTAU_TRACE:
			cmd = CMD_ALL_TRACE_SIZE;
			break;
		default:
			//def is profile
			cmd = CMD_ALL_SIZE;
		}

		//fd = open(path,O_RDONLY);	
		fd = open(path,O_RDONLY);	
		if(fd < 0)
		{
			perror("open:");
			return fd;
		}

		ktau_pkg.op.size = size;
		ktau_pkg.type = cmd;
		ktau_pkg.pid = -1;
		ktau_pkg.nopids = nopids;
		
		if(nopids == 0) { //then ALL processes
			hdrbuf = (char*)&ktau_pkg;
		} else { //nopids >= 1, then some other or many processes
			hdrbuf = (char*)malloc(sizeof(ktau_package_t) + (sizeof(pid_t) * nopids));
			*((ktau_package_t*)hdrbuf) = ktau_pkg;
			memcpy(hdrbuf+sizeof(ktau_package_t), pid, sizeof(pid_t)*nopids);
		}

		ret_value = 0;
		
		ret_value = ioctl(fd, cmd, hdrbuf);
		
		ktau_pkg = *((ktau_package_t*)hdrbuf);

		lc_size = ktau_pkg.op.size.size;
		if(flags)
			*flags = ktau_pkg.op.size.flags;

	}
	//else /* Many Pids: We dont handle it now */
	//{
	//	perror("read_size: Dont Handle Multiple PIDs now. Only Single & ALL.\n");
	//	ret_value = -1;
	//}

	close(fd);

	if(ret_value < 0)
	{
		perror("read:");
		return ret_value;
	}

	ret_value = 0;

	if(compratio == -1)
	{	
		compratio = 0.5;
	}

	lc_size = lc_size + (unsigned long)(lc_size * compratio);

	return lc_size;
}


/* read_data: Used to query /proc/ktau for DATA of profile(s). *
 ***************************************************************
 * Arguments:
 * type:        KTAU_PROFILE or KTAU_TRACE
 *
 * self:        if reading self -> set to 1, otherwise set to zero
 *
 * pid:         pointer to a single pid , or address of array of pids
 *              Can be NULL --> request data of entire system prof
 *
 * nopids:      no of pids being pointed to above
 *              Can be ZERO --> request data of entire system prof
 *
 * buffer:      pointer to allocated memory
 *
 * size:        size of buffer (allocate memory above)
 *
 * tol:         tolerance paramter. to ignore, set to ZERO
 *
 * flags:       <TO BE DONE> . ignore, set to NULL.
 *
 * Returns:
 * On Success:  total size of profile(s) data read into buffer.
 * On Error:    -1; Other negative error codes reserved.
 *
 * Constraints:
 * Must have called read_size before to ascertain size of buffer.
 *
 * NOTE: Even if SIZE allocated is that returned by read_size, 
 * read_data can return error for lack of size -> as size of prof
 * can change after read_size. This is unlikely.
 */
long read_data(int type, int self, pid_t *pid, unsigned int nopids, char* buffer, unsigned long size, unsigned long tol, unsigned int* flags)
{
	int ret_value = 0;
	int fd = -1;
	char path[MAX_KTAU_PROC_PATH+1];
	ktau_package_t ktau_pkg;
	ktau_read_t kbuffer;
	char *lc_buf = NULL;
	char* temp_buf = NULL;
	unsigned int in_size = sizeof(ktau_package_t) + size ;
	unsigned int cmd = 0;

	//initialize buffer with user-args
	kbuffer.tol.tol_in = tol;
	kbuffer.data = buffer;
	kbuffer.size = size;
	kbuffer.flags = 0;
	if(flags)
		kbuffer.flags = *flags;

	if( !((nopids == 1) && (self == 1)) && (size < (sizeof(pid_t) * nopids)) ){
		lc_buf = (char*)malloc(sizeof(ktau_package_t) + (sizeof(pid_t) * nopids));
	} else {
		lc_buf = (char*)malloc(in_size);
	}

	if(!lc_buf)
	{
		perror("lc_buf malloc: ");
		return -1;
	}

	switch(type) {
	case KTAU_PROFILE:
		snprintf(path, MAX_KTAU_PROC_PATH, PROFILE_PATH);
		break;
	case KTAU_TRACE:
		snprintf(path, MAX_KTAU_PROC_PATH, TRACE_PATH);
		break;
	default:
		//def is profile
		snprintf(path, MAX_KTAU_PROC_PATH, PROFILE_PATH);
	}

	if((nopids == 1)&& (self == 1)) /* single pid get read : use the read method */
	{
		switch(type) {
		case KTAU_PROFILE:
			cmd = CMD_READ;
			break;
		case KTAU_TRACE:
			cmd = CMD_TRACE_READ;
			break;
		default:
			//def is profile
			cmd = CMD_READ;
		}

		fd = open(path,O_RDONLY);	
		if(fd < 0)
		{
			perror("open:");
			free(lc_buf);
			return fd;
		}

		ktau_pkg.op.read= kbuffer; /* so that flags/tolerance passed from app is recorded */
		ktau_pkg.type = cmd;
		ktau_pkg.pid = *pid;
		ktau_pkg.nopids = nopids; 
		
		memcpy(lc_buf, &ktau_pkg, sizeof(ktau_pkg));

		ret_value = 0;
		/* CHANGED : To Use only ioctl
		 * do {
			ret_value = write(fd, lc_buf, sizeof(ktau_package_t) + kbuffer.size);
		} while((ret_value < 0) && (errno == EINTR));
		*/
		ret_value = ioctl(fd, cmd, lc_buf);

	}
	else //if(nopids == 0) /* ALL Pids : use IOCTL */
	{
		switch(type) {
		case KTAU_PROFILE:
			cmd = CMD_ALL_READ;
			break;
		case KTAU_TRACE:
			cmd = CMD_ALL_TRACE_READ;
			break;
		default:
			//def is profile
			cmd = CMD_ALL_READ;
		}

		fd = open(path,O_RDONLY);	
		if(fd < 0)
		{
			perror("open:");
			free(lc_buf);
			return fd;
		}

		ktau_pkg.op.read= kbuffer; /* So that flags/tolerance passed from app is recorded */
		ktau_pkg.type = cmd;
		ktau_pkg.nopids = nopids; 

		memcpy(lc_buf, &ktau_pkg, sizeof(ktau_pkg));

		if(nopids != 0) { //then NOT ALL , but one or many other processes
			memcpy(lc_buf+sizeof(ktau_package_t), pid, sizeof(pid_t)*nopids);
		}
		
		ret_value = ioctl(fd, cmd, lc_buf);

	}
	//else /* Mamny pids : We dont support */
	//{
	//	perror("read_data: Many PIDs not supported. Only Single PID & ALL Pids.");
	//	ret_value = -1;
	//}

	close(fd);

	if(ret_value < 0)
	{
		perror("read/ioctl ret val < 0:");
		if(lc_buf)
		{
			free(lc_buf);
		}
		return ret_value;
	}

	temp_buf = kbuffer.data;
	kbuffer = (*(ktau_package_t*)lc_buf).op.read;
	kbuffer.data = temp_buf;

	memcpy(kbuffer.data, (char*)(lc_buf + sizeof(ktau_package_t)), ret_value - sizeof(ktau_package_t));

	if(flags)
		*flags = kbuffer.flags;

	ret_value = 0;
	
	free(lc_buf);

	return kbuffer.size;
}


/* purge_data: Used to reset state of profiles /proc/ktau for pid(s). *
 **********************************************************************
 * Arguments:
 * pid:         pointer to a single pid , or address of array of pids
 *              Can be NULL --> request reset of entire system prof
 *
 * nopids:      no of pids being pointed to above
 *              Can be ZERO --> request reset of entire system prof
 *
 * tol:         tolerance paramter. to ignore, set to ZERO
 *
 * flags:       <TO BE DONE> . ignore, set to NULL.
 *
 * Returns:
 * On Success:  0
 * On Error:    -1; Other negative error codes reserved.
 *
 * Constraints:
 * None.
 */
long purge_data(pid_t *pid, unsigned int nopids, unsigned long tol, unsigned int* flags)
{
	int ret_value = 0;
	int fd = -1;
	char path[MAX_KTAU_PROC_PATH+1];
	ktau_package_t ktau_pkg;
	ktau_write_t kbuffer;
	char* lc_buf = NULL;
	char* temp_buf = NULL;

	//initialize buffer with user-args
	kbuffer.tol.tol_in = tol;
	kbuffer.flags = 0;
	if(flags)
		kbuffer.flags = *flags;

	lc_buf = (char*)malloc(sizeof(ktau_package_t));
	if(!lc_buf)
	{
		perror("malloc:");
		return -1;
	}

	if(nopids == 1) /* single pid write : use the write method */
	{
		/* CHANGED: TO use only all & ioctl 
		 * snprintf(path, MAX_KTAU_PROC_PATH, "/proc/ktau/%u", *pid);
		*/
		snprintf(path, MAX_KTAU_PROC_PATH, "/proc/ktau/all");

		fd = open(path,O_RDONLY);	
		if(fd < 0)
		{
			perror("open:");
			free(lc_buf);
			return fd;
		}

		ktau_pkg.op.write = kbuffer; /* so that flags/tolerance passed from app is recorded */
		ktau_pkg.type = CMD_PURGE;
		ktau_pkg.pid = *pid;
		
		memcpy(lc_buf, &ktau_pkg, sizeof(ktau_pkg));

		ret_value = 0;
		/* CHANGED: To use ioctl ONLY
		 * do {
			ret_value = write(fd, lc_buf, sizeof(ktau_package_t));
		} while((ret_value < 0) && (errno == EINTR));
		*/
		ret_value = ioctl(fd, CMD_PURGE, lc_buf);

		ktau_pkg = *((ktau_package_t*)lc_buf);
	}
	else if(nopids == 0) /* ALL Pids : use IOCTL */
	{
		snprintf(path, MAX_KTAU_PROC_PATH, "/proc/ktau/all");

		fd = open(path,O_RDONLY);	
		if(fd < 0)
		{
			perror("open:");
			free(lc_buf);
			return fd;
		}

		ktau_pkg.op.write = kbuffer; /* So that flags/ tolerance passed from app is recorded */
		ktau_pkg.type = CMD_ALL_PURGE;
		
		memcpy(lc_buf, &ktau_pkg, sizeof(ktau_pkg));

		ret_value = ioctl(fd, CMD_ALL_PURGE, lc_buf);

		ktau_pkg = *((ktau_package_t*)lc_buf);
	}
	else /*Many pids : not supported */
	{
		perror("pruge_data: Many PIDs not supported. Only Single & ALL Pids.");
		ret_value = -1;
	}

	close(fd);

	if(ret_value < 0)
	{
		perror("write / ioctl:");
		if(lc_buf)
		{
			free(lc_buf);
		}
		return ret_value;
	}

	kbuffer = ktau_pkg.op.write;

	if(flags)
		*flags = kbuffer.flags;

	ret_value = 0;
	
	free(lc_buf);

	return ret_value;
}


/* ktau_set_state : Used to query /proc/ktau for SIZE of profile(s). *
 ***************************************************************
 * Arguments:
 * pid:         pointer to a single pid (MUST BE SELF)
 *              Can be NULL (if pid is unknown, e.g. for threads) 
 *
 // state:	ptr to ptr to ktau_state (CAN be NULL, to Unset)
 * state:	PTR to ktau_state (CAN be NULL, to Unset) (buffer must have been atleats a page long
 *
 * flags:       <TO BE DONE> . ignore, set to NULL.
 *
 * Returns:
 * On Success:  Zero.
 * On Error:    -1; Other negative error codes reserved.
 *
 * Constraints:
 * ktau_set_state only works on 'SELF' , cannot perform it (yet) on 
 * other processes.
 */
long ktau_set_state(pid_t *pid, ktau_state* state, unsigned int* flags)
{
	int ret_value = 0;
	int fd = -1;
	char path[MAX_KTAU_PROC_PATH+1];
	ktau_package_t ktau_pkg;
	ktau_merge_t kbuffer;

	//initialize buffer with user-args
	kbuffer.flags = 0;
	if(flags)
		kbuffer.flags = *flags;

	kbuffer.ppstate = state;

	//snprintf(path, MAX_KTAU_PROC_PATH, "/proc/ktau/all");
	snprintf(path, MAX_KTAU_PROC_PATH, PROFILE_PATH);

	fd = open(path,O_RDONLY);	
	if(fd < 0)
	{
		perror("open:");
		return fd;
	}

	ktau_pkg.op.merge = kbuffer; /* so that flags/tolerance passed from app is recorded */
	ktau_pkg.type = CMD_SET_MERGE;
	if(pid) {
		ktau_pkg.pid = *pid;
	} else {
		ktau_pkg.pid = -1;
	}
	
	ret_value = ioctl(fd, CMD_SET_MERGE, (char*)(&ktau_pkg));

	close(fd);

	if(ret_value < 0)
	{
		perror("ioctl:");
	}
	else 
	{
		ret_value = 0;
	}

	return ret_value;
}


/* Once Read (into Buffer), that Data needs to be *
 * expanded into Profile Data.                    *
 * type:        KTAU_PROFILE or KTAU_TRACE
 *
 * Returns: no profiles expanded. -1 on error.
 */
long unpack_bindata(int type, char* buffer, unsigned long size, ktau_output** output) 
{
        int alloc_size = 0;
        int i = 0;
        char *bufptr = 0x0;
	int noprofs = -1;

	bufptr = buffer;

	memcpy(&noprofs, buffer, sizeof(unsigned int));	//<for no-of-profiles>
	bufptr+=sizeof(unsigned int);

	*output = (ktau_output*)malloc(noprofs * sizeof(ktau_output));
        if(!*output) {
                perror("malloc failed");
                return -1;
        }

        for(i = 0; i< noprofs; i++)
        {
		memcpy(&((*output)[i].pid), bufptr, sizeof(pid_t)); //<for pid>
		bufptr+=sizeof(unsigned int);
		memcpy(&((*output)[i].size), bufptr, sizeof(unsigned int)); //<for size>
		bufptr+=sizeof(unsigned int);

		if(type == KTAU_PROFILE) {

			(*output)[i].ent_lst = (o_ent*)malloc(sizeof(o_ent)*(*output)[i].size);
			if(!((*output)[i].ent_lst)) {
				perror("malloc failed");
				return -1;
			}
			memcpy((*output)[i].ent_lst, bufptr, sizeof(o_ent) * (*output)[i].size); //<for o_ent>
			bufptr+= (sizeof(o_ent) * (*output)[i].size);

		} else if(type == KTAU_TRACE) {

			(*output)[i].trace_lst = (ktau_trace*)malloc(sizeof(ktau_trace)*(*output)[i].size);
			if(!((*output)[i].trace_lst)) {
				perror("malloc failed");
				return -1;
			}
			memcpy((*output)[i].trace_lst, bufptr, sizeof(ktau_trace) * (*output)[i].size); //<for ktau_trace>
			bufptr+= (sizeof(ktau_trace) * (*output)[i].size);

		}
        }

        return noprofs;
}


/* Routine prints out an unpacked ktau_output       *
 * to the provided file-stream (can be stdout etc). *
 * Expects a const ptr to a single ktau_output */

/* Helper */
#ifdef __cplusplus
extern "C" {
#endif
static void print_o_ent(FILE* fp, o_ent* pent);
static void print_trace(FILE* fp, ktau_trace* ptrace);
#ifdef __cplusplus
}
#endif

 /* type:        KTAU_PROFILE or KTAU_TRACE
 */
void print_many_profiles(int type, FILE* fp, const ktau_output* profiles, unsigned int no_profiles)
{
	int i = 0;

	fprintf(fp, "No Profiles: %u\n", no_profiles);
	for(i=0; i< no_profiles; i++)
	{
		print_ktau_output(type, fp, &(profiles[i]));
	}
}

void print_ktau_output(int type, FILE* fp, const ktau_output* profile)
{
	o_ent* pent = NULL;
	ktau_trace* ptrace = NULL;
	int i = 0;

	if((!profile) || (!fp))
	{
		perror("Input Null ptr. \n");
		return;
	}

	fprintf(fp, "PID: %d\t No Entries: %u\n", profile->pid, profile->size);

	for(i=0; i<profile->size; i++)
	{
		if(type == KTAU_PROFILE) {

			pent = &(profile->ent_lst[i]);
			print_o_ent(fp, pent);

		} else if(type == KTAU_TRACE) {

			ptrace = &(profile->trace_lst[i]);
			print_trace(fp, ptrace);

		}
	}

}

/* helper to print o_ents */
static void print_o_ent(FILE* fp, o_ent* pent)
{
	h_ent* ph_ent = NULL;
	int i = 0;
	ktau_timer* timer;
	int index = 0;

	if(!fp || !pent)
	{
		perror("Null input. \n");
		return;
	}

	index = pent->index;

	ph_ent = &(pent->entry);

	if(!ph_ent)
	{
		fprintf(fp, "NULL\n");
		return;
	}

	timer = &(ph_ent->data.timer);

        //if(timer->count){
                fprintf(fp, "Entry %4d: addr %x, count %4u, incl %4llu, excl %4llu\n",
                                        index,
                                        ph_ent->addr,
                                        timer->count,
                                        timer->incl,
					timer->excl);
	//}

	return;
}

/* helper to print ktau_traces */
static void print_trace(FILE* fp, ktau_trace* ptrace)
{
	int i = 0;

	if(!fp || !ptrace)
	{
		perror("Null input. \n");
		return;
	}

        fprintf(fp, "%llu %x %u\n",
                                        ptrace->tsc,
                                        ptrace->addr,
                                        ptrace->type);

	return;
}


/*
long ktau_dump_toggle()
{
        int ret_value = 0;
        int fd = -1;
        char path[MAX_KTAU_PROC_PATH+1];
        ktau_package_t ktau_pkg;
        unsigned long lc_size = 0;

        snprintf(path, MAX_KTAU_PROC_PATH, "/proc/ktau/all");

        fd = open(path,O_RDONLY); 
        if(fd < 0)
        {
                perror("open:");
                return fd;
        }

        ktau_pkg.type = CMD_SET_DUMP;
        ret_value = 0;
        
        ret_value = ioctl(fd, CMD_SET_DUMP, &ktau_pkg);

        close(fd);

        if(ret_value < 0)
        {
                perror("ioctl:");
        }

        return ret_value;
}
*/


/* BinData From File                              *
 * expanded into Profile Data.                    *
 * Returns: no profiles expanded. -1 on error.    *
 * Contraint: Caller needs to free ktau_output mem*
 */
long unpack_bindata_file(int type, char* path, ktau_output** output)
{
        int size = 0, i = 0, fd = -1, noprofs = 0;
        long ret_value = -1;
        struct stat statbuf;
        char* buffer = NULL, *mv_buffer = NULL;

        if(!path) {
                return -1;
        }

        if(stat(path, &statbuf) != 0) {
                perror("stat ret error.\n");
                return -1;
        }

        size = statbuf.st_size;

        buffer = (char*)malloc(size);
        if(!buffer) {
                perror("malloc failed.\n");
                return -1;
                exit(-1);
        }

        fd = open(path, O_RDONLY);
        if(fd <= 0) {
                perror("open failed.\n");
                return -1;
                exit(-1);
        }

        ret_value = read(fd, buffer, size);
        if(ret_value < size) {
                perror("read: less than required.\n");
                //exit(-1);
        }

        mv_buffer = buffer + sizeof(ktau_package_t);

        /* Unpack Data*/
        noprofs = unpack_bindata(type, mv_buffer, size, output);

        free(buffer);

        return noprofs;
}

/* aggr_many_profiles: Aggregates array of Profiles to provide single kernel-as-a-whole profile 
 *****************************************************************
 * Arguments:
 * inprofiles:  ktau_output ptr to the Unpacked profile data.
 *
 * no_profiles: the number of profiles pointed to by profiles* (above)
 *
 * max_prof_entries: the maximum no of entries a single profile may have
 *
 * outprofile:  ptr to allocated memory for one profile with max_prof_size entries
 *
 * Returns:     0 on success, negative on error
 * 
 * Constraints:
 * Can aggregate from only ktau_output (i.e unpacked profiles). Cannot 
 * aggregate packed-binary profile data --> therefore unpack_bindata
 * must be called on those before calling this.
 */
int aggr_many_profiles(const ktau_output* inprofiles, unsigned int no_profiles, unsigned int max_prof_entries, ktau_output* outprofile)
{
        int ret_val = 0, profno = 0, entno = 0, cur_index = 0;
        const ktau_output* curr_prof = NULL;
        h_ent *out_hent, *cur_hent;
        o_ent * out_oent = NULL;
        int maxindex = 0;

        for(profno = 0; profno<no_profiles; profno++) {
                curr_prof = (inprofiles + profno);
                for(entno = 0; entno < curr_prof->size; entno++) {
                        cur_index = (curr_prof->ent_lst + entno)->index;
                        cur_hent = &((curr_prof->ent_lst + entno)->entry);
                        out_hent = &((outprofile->ent_lst + cur_index)->entry);
                        out_oent = (outprofile->ent_lst + cur_index);
                        out_oent->index = cur_index;
                        out_hent->addr = cur_hent->addr;
                        //assume data is timer for now!
                        if(cur_hent->type == KTAU_TIMER) { 
                                out_hent->data.timer.count += cur_hent->data.timer.count;
                                out_hent->data.timer.incl += cur_hent->data.timer.incl;
                                out_hent->data.timer.excl += cur_hent->data.timer.excl;
                        }
                }
                
                if(maxindex < cur_index) {
                        maxindex = cur_index;
                }
        }
        
        outprofile->size = maxindex;
        
        return ret_val;
}


/***************************************************************************
 * $RCSfile: ktau_proc_interface.c,v $   $Author: anataraj $
 * $Revision: 1.2 $   $Date: 2006/11/09 06:11:10 $
 * POOMA_VERSION_ID: $Id: ktau_proc_interface.c,v 1.2 2006/11/09 06:11:10 anataraj Exp $ 
 ***************************************************************************/


