#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <stdlib.h>

#include <android/log.h>

#include "adb.h"
#include "jdwp.h"

#define LOGV(...) //__android_log_print(ANDROID_LOG_VERBOSE, "TAU", __VA_ARGS__)

static unsigned local_id = 1;

static const char*
command_str(unsigned cmd)
{
    switch(cmd){
    case A_SYNC: return "SYNC";
    case A_CNXN: return "CNXN";
    case A_OPEN: return "OPEN";
    case A_OKAY: return "OKAY";
    case A_CLSE: return "CLSE";
    case A_WRTE: return "WRTE";
    case A_AUTH: return "AUTH";
    default:     return "????";
    }
}

#define DUMPMAX 128
static void
dump_message(const char *label, msg_t *msg)
{
    int i;
    const char *tag;
    char *x;
    unsigned count;

    tag = command_str(msg->command);

    LOGV("%s: %s %08x %08x %04x %08x\"",
	 label, tag, msg->arg0, msg->arg1, msg->data_length, msg->data_check);

    count = msg->data_length;
    x = (char*) msg->data;

    /*
    if(count > DUMPMAX) {
        count = DUMPMAX;
        tag = "";
    } else {
        tag = "\"";
    }
    for(i=0; i<count; i++) {
        if((*x >= ' ') && (*x < 127)) {
            fputc(*x, stderr);
        } else {
            fputc('.', stderr);
        }
        x++;
    }
    fputs(tag, stderr);

    printf(" ::dex:: ");
    x = (char*) msg->data;
    for(i=0; i<count; i++) {
	printf("%02x ", *x++);
    }
    */

#define H(m) ((m<c) ? x[16*r+m] : 0)
#define C(m) ((m<c) ? ((x[16*r+m]>=' ')&&(x[16*r+m]<127) ? x[16*r+m] : '.') : ' ')
    char *line;
    int r, c; // row, column
    for (r=0; r<(count+15)/16; r++) {
	c = (count - 16 * r) > 16 ? 16 : (count - 16 * r);

	asprintf(&line,
		 "\t"						      \
		 "%02x %02x %02x %02x %02x %02x %02x %02x    "	      \
		 "%02x %02x %02x %02x %02x %02x %02x %02x    "	      \
		 "| %c%c%c%c%c%c%c%c %c%c%c%c%c%c%c%c |",
		 H( 0), H( 1), H( 2), H( 3), H( 4), H( 5), H( 6), H( 7),
		 H( 8), H( 9), H(10), H(11), H(12), H(13), H(14), H(15),
		 C( 0), C( 1), C( 2), C( 3), C( 4), C( 5), C( 6), C( 7),
		 C( 8), C( 9), C(10), C(11), C(12), C(13), C(14), C(15));

	LOGV(line);

	free(line);
    }
    
    printf("\n");
}


static msg_t *
new_message(adb_ctx_t *ctx, unsigned cmd, unsigned arg0,
	    unsigned arg1, unsigned data_length, char *data)
{
    int i;
    msg_t *msg = &ctx->smsg;

    if (data_length > ctx->max_payload) {
	return NULL;
    }

    msg->command     = cmd;
    msg->arg0        = arg0;
    msg->arg1        = arg1;
    msg->data_length = data_length;
    msg->magic       = cmd ^ 0xffffffff;
    msg->data_check  = 0;

    for (i=0; i<data_length; i++) {
	msg->data_check += (unsigned)data[i];
    }

    memcpy(msg->data, data, data_length);

    return msg;
}

static int
send_message(adb_ctx_t *ctx)
{
    int rv;
    int finished, remain;
    msg_t *msg = &ctx->smsg;
    char *data = (char*)msg;

    dump_message(__func__, msg);

    finished = 0;
    remain   = sizeof(msg_t) + msg->data_length;

    while (remain > 0) {
	rv = write(ctx->fd, data + finished, remain);
	if (rv < 0) {
	    if (errno == EINTR) {
		continue;
	    } else {
		perror("write");
		return -1;
	    }
	}

	finished += rv;
	remain   -= rv;
    }

    return 0;
}

static msg_t *
recv_message(adb_ctx_t *ctx)
{
    int rv;
    int finished, remain;
    char *data;
    unsigned sum = 0;
    msg_t *msg = &ctx->rmsg;

    /* get message header */
    data     = (char*)msg;
    finished = 0;
    remain   = sizeof(msg_t);
    while (remain > 0) {
	rv = read(ctx->fd, data + finished, remain);
	if (rv < 0) {
	    if (errno == EINTR) {
		continue;
	    } else {
		perror("read");
		return NULL;
	    }
	}

	finished += rv;
	remain   -= rv;
    }

    /* get payload, if any */
    data     = msg->data;
    finished = 0;
    remain   = msg->data_length;
    if (remain > ctx->max_payload) { // this shouldn't happen
	return NULL;
    }
    while (remain > 0) {
	rv = read(ctx->fd, data + finished, remain);
	if (rv < 0) {
	    if (errno == EINTR) {
		continue;
	    } else {
		perror("read");
		return NULL;
	    }
	}

	finished += rv;
	remain   -= rv;
    }

    dump_message(__func__, msg);

    /* sanity check */
    if ((msg->command ^ 0xffffffff) != msg->magic) {
	return NULL;
    }
    while (remain != finished) {
	sum += (unsigned)data[remain++];
    }
    if (sum != msg->data_check) {
	return NULL;
    }

    return msg;
}

static int
adb_send_connect(adb_ctx_t *ctx)
{
    char id_string[] = "host::";

    new_message(ctx,
		A_CNXN, A_VERSION, MAX_PAYLOAD, sizeof(id_string), id_string);

    return send_message(ctx);
}

static int
adb_send_open(adb_ctx_t *ctx, pid_t pid)
{
    char dst[16]; // this should be enough 

    snprintf(dst, sizeof(dst), "jdwp:%d", pid);

    new_message(ctx, A_OPEN, ctx->lid, 0, sizeof(dst), dst);

    return send_message(ctx);
}

static int
adb_send_write(adb_ctx_t *ctx, char *data, int len)
{
    new_message(ctx, A_WRTE, ctx->lid, ctx->rid, len, data);

    return send_message(ctx);
}

static int
adb_send_okay(adb_ctx_t *ctx)
{
    new_message(ctx, A_OKAY, ctx->lid, ctx->rid, 0, NULL);

    return send_message(ctx);
}


static int
rmsg_is_cnxn(adb_ctx_t *ctx)
{
    msg_t *msg = &ctx->rmsg;

    if (msg->command != A_CNXN) {
	return 0;
    }

    if (msg->arg0 != A_VERSION) {
	fprintf(stderr, "Error: ADB: CNXN: version mismatch.\n");
	return 0;
    }

    if (msg->arg1 > ctx->max_payload) {
	fprintf(stderr, "Error: ADB: CNXN: max_payload exceed limit.\n");
	return 0;
    }

    return 1;
}

static int
_id_is_ok(adb_ctx_t *ctx, char *tag)
{
    msg_t *msg = &ctx->rmsg;

    if ((ctx->rid != 0) && (msg->arg0 != ctx->rid)) {
	fprintf(stderr, "Error: ADB: %s: remote_id should be %d, get %d\n",
		tag, ctx->rid, msg->arg0);
	return 0;
    }

    if (msg->arg1 != ctx->lid) {
	fprintf(stderr, "Error: ADB: %s: local_id should be %d, get %d\n",
		tag, ctx->lid, msg->arg1);
	return 0;
    }

    return 1;
}

static int
rmsg_is_wrte(adb_ctx_t *ctx)
{
    msg_t *msg = &ctx->rmsg;

    if (msg->command != A_WRTE) {
	return 0;
    }

    return _id_is_ok(ctx, "WRTE");
}

static int
rmsg_is_okay(adb_ctx_t *ctx)
{
    msg_t *msg = &ctx->rmsg;

    if (msg->command != A_OKAY) {
	return 0;
    }

    return _id_is_ok(ctx, "OKAY");
}

static int
rmsg_is_clse(adb_ctx_t *ctx)
{
    msg_t *msg = &ctx->rmsg;

    if (msg->command != A_CLSE) {
	return 0;
    }

    return 1;
}

static int
adb_backlog(adb_ctx_t *ctx, msg_t *msg)
{
    adb_backlog_t *backlog;

    backlog = (adb_backlog_t*)malloc(sizeof(*backlog) +
				     msg->data_length);
    if (backlog == NULL) {
	return -1;
    }

    backlog->len    = msg->data_length;
    backlog->offset = 0;
    memcpy(backlog->data, msg->data, backlog->len);

    if (ctx->backlog == NULL) {
	ctx->backlog  = backlog;
	backlog->next = backlog;
	backlog->prev = backlog;
    } else {
	backlog->next       = ctx->backlog;
	backlog->prev       = ctx->backlog->prev;
	backlog->next->prev = backlog;
	backlog->prev->next = backlog;
    }

    return 0;
}

static int
adb_free_backlog(adb_ctx_t *ctx)
{
    adb_backlog_t *this, *next;

    /* free any backlog which remains unhandled */
    if (ctx->backlog != NULL) {
	this = ctx->backlog;
	do {
	    next = this->next;
	    free(this);
	    this = next;
	} while (this != ctx->backlog);
    }

    return 0;
}

/*****************************************************/

adb_ctx_t*
adb_open(pid_t pid)
{
    int rv;
    msg_t *msg;
    adb_ctx_t *ctx;
    struct sockaddr_in saddr;

    ctx = (adb_ctx_t*)malloc(sizeof(*ctx));
    if (ctx == NULL) {
	perror("malloc");
	return NULL;
    }

    saddr.sin_family      = AF_INET;
    saddr.sin_port        = htons(ADBD_PORT);
    saddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    ctx->fd          = socket(AF_INET, SOCK_STREAM, 0);
    ctx->lid         = local_id++;
    ctx->rid         = 0;
    ctx->max_payload = MAX_PAYLOAD;
    ctx->backlog     = NULL;

    rv = connect(ctx->fd, (struct sockaddr*)&saddr, sizeof(saddr));
    if (rv < 0) {
	LOGV(" *** Error: ADB: adb_open: %s", strerror(errno));
	goto err_q_1;
    }

    /*
     * adb protocol init sequence:
     *
     *       adb         adbd
     *      =================
     *  1.  CNXN  ----->
     *  2.        <----- CNXN
     *  3.  OPEN  ----->
     *  4.        <----- OKAY
     */

    /* STEP 1. */
    rv = adb_send_connect(ctx);
    if (rv < 0) {
	goto err_q_1;
    }

    /* STEP 2. */
    msg = recv_message(ctx);
    if (msg == NULL) {
	goto err_q_1;
    }

    if (rmsg_is_cnxn(ctx)) {
	/* update max_payload limitation */
	ctx->max_payload = msg->arg1;
    } else {
	LOGV(" *** Error: ADB: need CNXN, get %s\n",
		command_str(msg->command));
	goto err_q_1;
    }
    
    /* STEP 3. */
    rv = adb_send_open(ctx, pid);
    if (rv < 0) {
	goto err_q_1;
    }

    /* STEP 4. */
    msg = recv_message(ctx);
    if (msg == NULL) {
	goto err_q_1;
    }

    if (rmsg_is_okay(ctx)) {
	/* update remote id */
	ctx->rid = msg->arg0;
    } else if (rmsg_is_clse(ctx)) {
	LOGV(" *** Error: ADB: connection closed by peer.\n");
	goto err_q_1;
    } else {
	LOGV(" *** Error: ADB: need OKAY, get %s\n",
		command_str(msg->command));
	goto err_q_1;
    }

    /* now the connection is considered as active */
    ctx->active = 1;

    return ctx;

err_q_1:
    close(ctx->fd);
    free(ctx);
    return NULL;
}

void
adb_close(adb_ctx_t *ctx)
{
    if (ctx) {
	adb_free_backlog(ctx);
	close(ctx->fd);

	free(ctx);
    }
}

int
adb_is_active(adb_ctx_t *ctx)
{
    return (ctx->active != 0);
}

ssize_t
adb_read(adb_ctx_t *ctx, char *buf, size_t count)
{
    int rv;
    msg_t *msg;
    int finished, remain, trunk;
    adb_backlog_t *backlog;

    finished = 0;
    remain   = count;

    while (remain > 0) {
	if (ctx->backlog != NULL) {
	    backlog = ctx->backlog;

	    if (remain > backlog->len - backlog->offset) {
		trunk = backlog->len - backlog->offset;
	    } else {
		trunk = remain;
	    }

	    memcpy(buf+finished, backlog->data + backlog->offset, trunk);

	    finished        += trunk;
	    remain          -= trunk;
	    backlog->offset += trunk;

	    /* remove the backlog if the data has all been handled */
	    if (backlog->len - backlog->offset == 0) {
		if (backlog->next == backlog) {
		    ctx->backlog = NULL;
		} else {
		    ctx->backlog->next->prev = ctx->backlog->prev;
		    ctx->backlog->prev->next = ctx->backlog->next;
		    ctx->backlog             = ctx->backlog->next;
		}

		free(backlog);
	    }
	} else {
	    /* bail out if connection is closed and no backlog remain */
	    if (!ctx->active) {
		return -1;
	    }

	    msg = recv_message(ctx);
	    if (msg == NULL) {
		return -1;
	    }

	    /* peer may close connection at anytime */
	    if (rmsg_is_clse(ctx)) {
		fprintf(stderr, "Error: ADB: connection closed by peer!\n");
		ctx->active = 0;
		return -1;
	    }

	    if (!rmsg_is_wrte(ctx)) {
		return -1;
	    }

	    /* update backlog, next iteration will consume it */
	    adb_backlog(ctx, msg);

	    /*
	     * If the peer doesn't receive our ack, they will stop sending us
	     * any data, so let's be careful. At least we should retry...
	     */
	    int retry = 3;
	    while (retry--) {
		rv = adb_send_okay(ctx);
		if (rv == 0) {
		    break;
		} else {
		    /* retry */
		}
	    }

	    /* something REALLY BAD has happened... */
	    if (rv < 0) {
		adb_free_backlog(ctx);
		return -1;
	    }
	}
    }

    return count;
}

ssize_t
adb_write(adb_ctx_t *ctx, char *buf, size_t count)
{
    int rv;
    msg_t *msg;
    int finished, remain, trunk;

    finished = 0;
    remain   = count;

    if (!ctx->active) {
	return -1;
    }

    while (remain > 0) {
	trunk = remain>ctx->max_payload ? ctx->max_payload : remain;

	rv = adb_send_write(ctx, buf+finished, trunk);
	if (rv < 0) {
	    return -1;
	}

	finished += trunk;
	remain   -= trunk;

	/* stay until we get our ack */
	while (1) {
	    msg = recv_message(ctx);
	    if (msg == NULL) {
		return -1;
	    }

	    /* nice, we get the ack */
	    if (rmsg_is_okay(ctx)) {
		break;
	    }

	    /* peer may close the connection at anytime */
	    if (rmsg_is_clse(ctx)) {
		fprintf(stderr, "Error: ADB: connection closed by peer!\n");
		ctx->active = 0;
		return -1;
	    }

	    /* peer may send us some data before the ack */
	    if (rmsg_is_wrte(ctx)) {
		adb_send_okay(ctx);
		adb_backlog(ctx, msg);
		continue;
	    }

	    /* that's all the posibilities above, we shouldn't reach here */
	}
    }

    return finished;
}

/*****************************************************/
