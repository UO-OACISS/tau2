#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <string.h>

#include "adb.h"
#include "jdwp.h"

static int jdwp_cmd_id = 0;

int
jdwp_handshake(jdwp_ctx_t *ctx)
{
    char shake[] = HANDSHAKE;
    char response[sizeof(HANDSHAKE)-1];

    adb_ctx_t *adb = ctx->adb;

    adb_write(adb, shake, sizeof(HANDSHAKE)-1);
    adb_read(adb, response, sizeof(HANDSHAKE)-1);

    if (memcmp(shake, response, sizeof(response)) != 0) {
	fprintf(stderr, "Error: JDWP: Handshake failed.\n");
	return -1;
    }

    return 0;
}

int
jdwp_send_pkt(jdwp_ctx_t *ctx, short cmd, char *data, int len)
{
    static int pkt_len = 0;
    static jdwp_cmd_t *pkt = NULL;

    adb_ctx_t *adb = ctx->adb;

    if (pkt_len < sizeof(jdwp_cmd_t) + len) {
	free(pkt); // free(NULL) has no effect

	pkt_len = sizeof(jdwp_cmd_t) + len;

	pkt = (jdwp_cmd_t*)malloc(pkt_len);
	if (pkt == NULL) {
	    pkt_len = 0;
	    return -1;
	}
    }

    pkt->length  = htonl(sizeof(jdwp_cmd_t) + len);
    pkt->id      = htonl(jdwp_cmd_id++);
    pkt->flags   = 0;
    pkt->cmd_set = (cmd & 0xff00) >> 8;
    pkt->command = (cmd & 0x00ff) >> 0;

    memcpy(pkt->data, data, len);

    return adb_write(adb, (char*)pkt, sizeof(jdwp_cmd_t)+len);
}

char *
jdwp_recv_pkt(jdwp_ctx_t *ctx)
{
    int rv;
    int data_len;
    char *pkt;
    jdwp_cmd_t header;
    jdwp_reply_t *reply = (jdwp_reply_t*)&header;

    adb_ctx_t *adb = ctx->adb;

    rv = adb_read(adb, (char*)&header, sizeof(jdwp_cmd_t));
    if (rv < 0) {
	return NULL;
    }

    /* fix byte order */
    header.length = ntohl(header.length);
    header.id     = ntohl(header.id);
    if (header.flags == 0x80) {
	reply->error_code = ntohl(reply->error_code);
    }

    pkt = (char*)malloc(header.length);

    memcpy(pkt, &header, sizeof(header));

    data_len = header.length - sizeof(header);

    if (data_len > 0) {
	rv = adb_read(adb, pkt+sizeof(header), data_len);
	if (rv < 0) {
	    free(pkt);
	    return NULL;
	}
    }

    return pkt;
}

int
jdwp_event_backlog(jdwp_ctx_t *ctx, jdwp_cmd_t *cmd)
{
    jdwp_event_t *event;

    event = (jdwp_event_t*)malloc(sizeof(jdwp_event_t));
    if (event == NULL) {
	return -1;
    }

    event->cmd = cmd;

    if (ctx->events == NULL) {
	ctx->events       = event;
	event->next       = event;
	event->prev       = event;
    } else {
	event->next       = ctx->events;
	event->prev       = ctx->events->prev;
	event->next->prev = event;
	event->prev->next = event;
    }

    return 0;
}

int
jdwp_init(jdwp_ctx_t *ctx)
{
    ctx->events = NULL;
    ctx->adb    = adb_open(getpid());

    if (ctx->adb == NULL) {
	return -1;
    }

    jdwp_handshake(ctx);

    return 0;
}

jdwp_reply_t *
jdwp_get_reply(jdwp_ctx_t *ctx)
{
    jdwp_reply_t *reply;

    while (1) {
	reply = (jdwp_reply_t*)jdwp_recv_pkt(ctx);
	if (reply == NULL) {
	    /* why? peer closed connection? */
	    if (!adb_is_active(ctx->adb)) {
		fprintf(stderr, "Error: JDWP: connection closed!\n");
		return NULL;
	    } else {
		/* retry */
		continue;
	    }
	}

	switch (reply->flags) {
	case 0x00:      /* jdwp events may come in anytime */
	    jdwp_event_backlog(ctx, (jdwp_cmd_t*)reply);
	    break;
	case 0x80:	/* okay, we get an reply */
	    return reply;
	default:        /* this can not happen */
	    fprintf(stderr, "Error: JDWP: malformed pkt, flags=0x%x\n", reply->flags);
	    free(reply);
	    break;
	}
    }

    return NULL;
}

int
jdwp_get_vm_version(jdwp_ctx_t *ctx)
{
    int rv;
    jdwp_reply_t *reply;

    rv = jdwp_send_pkt(ctx, VIRTUALMACHINE_VERSION, NULL, 0);
    if (rv < 0) {
	return -1;
    }

    reply = jdwp_get_reply(ctx);
    if (reply == NULL) {
	return -1;
    }

    if (reply->error_code != 0) {
	fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
		reply->error_code);
	free(reply);
	return -1;
    }

    printf("%s\n", reply->data+4);
    free(reply);
    return 0;
}

int
jdwp_set_event_request(jdwp_ctx_t *ctx, char eventKind, char suspendPolicy)
{
    int rv;
    int requestID;
    jdwp_reply_t *reply;

    char data[] = {eventKind, suspendPolicy, 0, 0, 0, 0};

    rv = jdwp_send_pkt(ctx, EVENTREQUEST_SET, data, sizeof(data));
    if (rv < 0) {
	return -1;
    }

    reply = jdwp_get_reply(ctx);
    if (reply == NULL) {
	return -1;
    }

    if ((reply->error_code != 0) || (reply->length != 15)) {
	fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
		reply->error_code);
	free(reply);
	return -1;
    }

    memcpy(&requestID, reply->data, 4);

    free(reply);

    /* the byte order of requestID doesn't matter */
    return requestID;
}

int
jdwp_resume_thread(jdwp_ctx_t *ctx, uint64_t threadID)
{
    int rv;
    jdwp_reply_t *reply;

    rv = jdwp_send_pkt(ctx, THREADREF_RESUME, (char*)&threadID, sizeof(threadID));
    if (rv < 0) {
	return -1;
    }

    reply = jdwp_get_reply(ctx);
    if (reply == NULL) {
	return -1;
    }

    if (reply->error_code != 0) {
	fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
		reply->error_code);
	free(reply);
	return -1;
    }

    free(reply);
    return 0;
}

char *
jdwp_get_thread_name(jdwp_ctx_t *ctx, uint64_t threadID)
{
    int rv;
    jdwp_reply_t *reply;

    rv = jdwp_send_pkt(ctx, THREADREF_NAME, (char*)&threadID, sizeof(threadID));
    if (rv < 0) {
	return NULL;
    }

    reply = jdwp_get_reply(ctx);
    if (reply == NULL) {
	return NULL;
    }

    if (reply->error_code != 0) {
	fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
		reply->error_code);
	free(reply);
	return NULL;
    }

    /* reply->data[]: 4-byte length followed by a non-NULL-terminated string */
    int len = ntohl(*(int*)(reply->data));
    char *name = (char*)malloc(len + 1);
    if (name == NULL) {
	free(reply);
	return NULL;
    }

    memcpy(name, reply->data+4, len);
    name[len] = 0;

    free(reply);
    return name;
}

uint64_t
jdwp_get_thread_group(jdwp_ctx_t *ctx, uint64_t threadID)
{
    int rv;
    jdwp_reply_t *reply;
    uint64_t threadGroupID;

    rv = jdwp_send_pkt(ctx, THREADREF_THREADGROUP, (char*)&threadID, sizeof(threadID));
    if (rv < 0) {
	return -1;
    }

    reply = jdwp_get_reply(ctx);
    if (reply == NULL) {
	return -1;
    }

    if (reply->error_code != 0) {
	fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
		reply->error_code);
	free(reply);
	return -1;
    }

    threadGroupID =  *(uint64_t*)(reply->data);

    free(reply);

    return threadGroupID;
}

char *
jdwp_get_thread_group_name(jdwp_ctx_t *ctx, uint64_t threadGroupID)
{
    int rv;
    jdwp_reply_t *reply;

    rv = jdwp_send_pkt(ctx, THREADGRPREF_NAME, (char*)&threadGroupID,
		       sizeof(threadGroupID));
    if (rv < 0) {
	return NULL;
    }

    reply = jdwp_get_reply(ctx);
    if (reply == NULL) {
	return NULL;
    }

    if (reply->error_code != 0) {
	fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
		reply->error_code);
	free(reply);
	return NULL;
    }
    /* reply->data[]: 4-byte length followed by a non-NULL-terminated string */
    int len = ntohl(*(int*)(reply->data));
    char *name = (char*)malloc(len + 1);
    if (name == NULL) {
	free(reply);
	return NULL;
    }

    memcpy(name, reply->data+4, len);
    name[len] = 0;

    free(reply);
    return name;
}

