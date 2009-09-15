#ifndef __TAU_VAMPIRTRACE_H__
#define __TAU_VAMPIRTRACE_H__

#ifdef TAU_64BITTYPES_NEEDED
#include <Profile/vt_inttypes.h>
#endif /* TAU_64BITTYPES_NEEDED */

#ifndef VT_NO_ID
#define VT_NO_ID                  0xFFFFFFFF
#endif /* VT_NO_ID */

#ifndef VT_NO_LNO
#define VT_NO_LNO                 0xFFFFFFFF
#endif /* VT_NO_LNO */

#ifndef VT_FUNCTION
#define VT_FUNCTION               1
#endif /* VT_FUNCTION */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void vt_open(void);
void vt_close(void);

#ifdef TAU_VAMPIRTRACE_5_7_API
uint32_t vt_def_region(const char* rname,
		       const char* rdesc,
		       uint32_t fid,
		       uint32_t begln,
		       uint32_t endln,
		       uint8_t rtype );
#define TAU_VT_DEF_REGION(rname,fid,begln,endln,rdesc,rtype) vt_def_region (rname, rdesc, fid, begln, endln, rtype);  
#else
uint32_t vt_def_region(const char* rname,
		       uint32_t fid,
		       uint32_t begln,
		       uint32_t endln,
		       const char* rdesc,
		       uint8_t rtype );
#define TAU_VT_DEF_REGION(rname,fid,begln,endln,rdesc,rtype) vt_def_region (rname, fid, begln, endln, rdesc, rtype);  
#endif

void vt_enter(uint64_t* time, uint32_t rid);
  
void vt_exit(uint64_t* time);
uint64_t vt_pform_wtime(void);

uint32_t vt_def_counter_group(const char* gname );
void vt_count(uint64_t* time, uint32_t cid, uint64_t cval);
uint32_t vt_def_counter(const char* cname, uint32_t cprop, uint32_t gid,
			const char* cunit);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __TAU_VAMPIRTRACE_H__ */
