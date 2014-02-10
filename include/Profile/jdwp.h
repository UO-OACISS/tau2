#ifndef _JDWP_H_
#define _JDWP_H_

#include "adb.h"

#define HANDSHAKE "JDWP-Handshake"

/* JDWP command definition: (command_set << 8) | command */
#define VIRTUALMACHINE_VERSION 0x0101
#define EVENTREQUEST_SET       0x0f01
#define EVENT_COMPOSIT         0x4064
#define THREADREF_RESUME       0x0b03
#define THREADREF_NAME         0x0b01

/* EventKind Constants */
#define E_THREAD_START 6
#define E_THREAD_END   7
#define E_VM_START     90
#define E_VM_DEATH     99

typedef struct __attribute__((packed)) {
    unsigned int length;
    unsigned int id;
    char         flags;
    char         cmd_set;
    char         command;
    char         data[];
} jdwp_cmd_t;

typedef struct __attribute__((packed)) {
    unsigned int length;
    unsigned int id;
    char         flags;
    short        error_code;
    char         data[];
} jdwp_reply_t;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int jdwp_handshake(adb_ctx_t *ctx);
int jdwp_read_events(adb_ctx_t *ctx);
int jdwp_resume_thread(adb_ctx_t *ctx, long long *threadID);
int jdwp_set_event_request(adb_ctx_t *ctx, char eventKind, char suspendPolicy);
int jdwp_get_vm_version(adb_ctx_t *ctx);

int jdwp_send_pkt(adb_ctx_t *ctx, short cmd, char *data, int len);
char *jdwp_recv_pkt(adb_ctx_t *ctx);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
