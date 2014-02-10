#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "adb.h"
#include "jdwp.h"

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

    fprintf(stderr, "%s: %s %08x %08x %04x %08x\"",
            label, tag, msg->arg0, msg->arg1, msg->data_length, msg->data_check);
    count = msg->data_length;
    x = (char*) msg->data;
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
    char id_string[] = "host:TAU:thread_event_monitor";

    new_message(ctx,
		A_CNXN, A_VERSION, MAX_PAYLOAD, sizeof(id_string), id_string);

    return send_message(ctx);
}

static int
adb_send_open(adb_ctx_t *ctx, pid_t pid)
{
    char dst[16];

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

    return _id_is_ok(ctx, "CLSE");
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

    rv = connect(ctx->fd, (struct sockaddr*)&saddr, sizeof(saddr));
    if (rv < 0) {
	perror("connect");
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
	fprintf(stderr, "Error: ADB: need CNXN, get %s\n",
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
	ctx->rid    = msg->arg0;
    } else if (rmsg_is_clse(ctx)) {
	fprintf(stderr, "Error: ADB: connection closed by peer.\n");
	goto err_q_1;
    } else {
	fprintf(stderr, "Error: ADB: need OKAY, get %s\n",
		command_str(msg->command));
	goto err_q_1;
    }

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
	close(ctx->fd);
	free(ctx);
    }
}

ssize_t
adb_read(adb_ctx_t *ctx, char *buf, size_t count)
{
    int rv;
    msg_t *msg;
    int finished, remain, trunk, offset;

    finished = 0;
    remain   = count;

    while (remain > 0) {
	if (ctx->backlog > 0) {
	    msg = &ctx->rmsg;
	    trunk  = remain>ctx->backlog? ctx->backlog : remain;
	    offset = msg->data_length - ctx->backlog;

	    memcpy(buf+finished, msg->data+offset, trunk);

	    finished     += trunk;
	    remain       -= trunk;
	    ctx->backlog -= trunk;
	} else {
	    msg = recv_message(ctx);
	    if (msg == NULL) {
		return -1;
	    }

	    if (!rmsg_is_wrte(ctx)) {
		return -1;
	    }

	    /* update backlog, next iteration will consume it */
	    ctx->backlog = msg->data_length;

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
		ctx->backlog = 0;
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

    while (remain > 0) {
	trunk = remain>ctx->max_payload ? ctx->max_payload : remain;

	rv = adb_send_write(ctx, buf+finished, trunk);
	if (rv < 0) {
	    return -1;
	}

	finished += trunk;
	remain   -= trunk;

	msg = recv_message(ctx);
	if (msg == NULL) {
	    return -1;
	}

	if (!rmsg_is_okay(ctx)) {
	    return -1;
	}
    }

    return count;
}

/*****************************************************/

/*
int
main(void)
{
    int rv;
    int threadStartReq, threadEndReq;
    msg_t *msg;
    struct sockaddr_in saddr;

    saddr.sin_family      = AF_INET;
    saddr.sin_port        = htons(ADBD_PORT);
    saddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    adb_fd = socket(AF_INET, SOCK_STREAM, 0);
    rv = connect(adb_fd, (struct sockaddr*)&saddr, sizeof(saddr));
    if (rv < 0) {
	perror("connect");
	return -1;
    }

    msg = (msg_t*)buf;

    adb_send_connect();
    recv_message(msg);

    adb_send_open();
    remote_id = adb_recv_okay();

    adb_send_okay();

    jdwp_handshake();
    jdwp_get_vm_version();

    threadStartReq = jdwp_set_event_request(6);
    if (threadStartReq < 0) {
	fprintf(stderr, "Failed to request thread events: requestID=0x%08x\n", threadStartReq);
	return -1;
    }
    threadEndReq = jdwp_set_event_request(7);
    if (threadEndReq < 0) {
	fprintf(stderr, "Failed to request thread events: requestID=0x%08x\n", threadEndReq);
	return -1;
    }

    jdwp_read_events();

    jdwp_clear_event_request(6, threadStartReq);
    jdwp_clear_event_request(7, threadEndReq);

    return 0;
}
*/
