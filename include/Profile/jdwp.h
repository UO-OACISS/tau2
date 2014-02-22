#ifndef _JDWP_H_
#define _JDWP_H_

#include "adb.h"

#define HANDSHAKE "JDWP-Handshake"

/* JDWP command definition: (command_set << 8) | command */
#define VIRTUALMACHINE_VERSION 0x0101
#define EVENTREQUEST_SET       0x0f01
#define EVENT_COMPOSIT         0x4064
#define THREADREF_NAME         0x0b01
#define THREADREF_RESUME       0x0b03
#define THREADREF_THREADGROUP  0x0b05
#define THREADGRPREF_NAME      0x0c01

/* EventKind Constants */
#define E_THREAD_START 6
#define E_THREAD_END   7
#define E_VM_START     90
#define E_VM_DEATH     99

/* SuspendPolicy Constants */
#define SUSPEND_NONE         0
#define SUSPEND_EVENT_THREAD 1
#define SUSPEND_ALL          2

typedef struct __attribute__((packed)) {
    unsigned int  length;
    unsigned int  id;
    unsigned char flags;
    char          cmd_set;
    char          command;
    char          data[];
} jdwp_cmd_t;

typedef struct __attribute__((packed)) {
    unsigned int  length;
    unsigned int  id;
    unsigned char flags;
    short         error_code;
    char          data[];
} jdwp_reply_t;

typedef struct _jdwp_event_t {
    jdwp_cmd_t *cmd;
    struct _jdwp_event_t *next;
    struct _jdwp_event_t *prev;
} jdwp_event_t;

typedef struct {
    adb_ctx_t *adb;
    jdwp_event_t *events;
} jdwp_ctx_t;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int jdwp_init(jdwp_ctx_t *ctx);
int jdwp_handshake(jdwp_ctx_t *ctx);
int jdwp_resume_thread(jdwp_ctx_t *ctx, uint64_t threadID);
char *jdwp_get_thread_name(jdwp_ctx_t *ctx, uint64_t threadID);
int jdwp_set_event_request(jdwp_ctx_t *ctx, char eventKind, char suspendPolicy);
int jdwp_get_vm_version(jdwp_ctx_t *ctx);
int jdwp_event_backlog(jdwp_ctx_t *ctx, jdwp_cmd_t *cmd);
int jdwp_send_pkt(jdwp_ctx_t *ctx, short cmd, char *data, int len);
char *jdwp_recv_pkt(jdwp_ctx_t *ctx);
uint64_t jdwp_get_thread_group(jdwp_ctx_t *ctx, uint64_t threadID);
char *jdwp_get_thread_group_name(jdwp_ctx_t *ctx, uint64_t threadGroupID);

jdwp_reply_t *jdwp_get_reply(jdwp_ctx_t *ctx);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
