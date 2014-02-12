#ifndef _ADB_H_
#define _ADB_H_

#include <sys/types.h>

#define ADBD_PORT 5555

#define A_VERSION   0x01000000  // ADB protocol version
#define MAX_PAYLOAD 4096

#define A_SYNC 0x434e5953
#define A_CNXN 0x4e584e43
#define A_OPEN 0x4e45504f
#define A_OKAY 0x59414b4f
#define A_CLSE 0x45534c43
#define A_WRTE 0x45545257
#define A_AUTH 0x48545541

/* see adb/protocol.txt */
typedef struct {
    unsigned command;       /* command identifier constant       */
    unsigned arg0;          /* first argument                    */
    unsigned arg1;          /* second argument                   */
    unsigned data_length;   /* length of payload (0 is allowed)  */
    unsigned data_check;    /* checksum of data payload          */
    unsigned magic;         /* command ^ 0xffffffff              */
    char     data[];
} msg_t;

typedef struct _adb_backlog_t {
    struct _adb_backlog_t *next;
    struct _adb_backlog_t *prev;

    int len;                /* data length                       */
    int offset;             /* read pointer offset of data       */
    char data[];
} adb_backlog_t;

typedef struct {
    int fd;                 /* socket fd                         */

    int lid;                /* local id                          */
    int rid;                /* remote id                         */
    int max_payload;        /* max payload size                  */

    adb_backlog_t *backlog; /* rmsg payload remain unhandled     */

    int active;             /* the connection is active or not   */

    msg_t rmsg;             /* the last received message         */
    char  d1[MAX_PAYLOAD];  /* aka. rmsg.data                    */
    msg_t smsg;             /* the last sent message             */
    char  d2[MAX_PAYLOAD];  /* aka. smsg.data                    */
} adb_ctx_t;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

adb_ctx_t* adb_open(pid_t pid);
void adb_close(adb_ctx_t *ctx);
int adb_is_active(adb_ctx_t *ctx);
ssize_t adb_read(adb_ctx_t *ctx, char *buf, size_t count);
ssize_t adb_write(adb_ctx_t *ctx, char *buf, size_t count);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
