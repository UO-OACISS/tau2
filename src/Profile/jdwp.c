#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>

#include "adb.h"
#include "jdwp.h"

static int jdwp_cmd_id = 0;

int
jdwp_handshake(adb_ctx_t *ctx)
{
    int rv;
    char shake[] = HANDSHAKE;
    char response[sizeof(HANDSHAKE)-1];

    adb_write(ctx, shake, sizeof(HANDSHAKE)-1);
    adb_read(ctx, response, sizeof(HANDSHAKE)-1);

    if (memcmp(shake, response, sizeof(response)) != 0) {
	fprintf(stderr, "Error: JDWP: Handshake failed.\n");
	return -1;
    }

    return 0;
}

int
jdwp_send_pkt(adb_ctx_t *ctx, short cmd, char *data, int len)
{
    static int pkt_len = 0;
    static jdwp_cmd_t *pkt = NULL;

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

    /* TODO: check length */
    memcpy(pkt->data, data, len);


    return adb_write(ctx, (char*)pkt, sizeof(jdwp_cmd_t)+len);
}

char *
jdwp_recv_pkt(adb_ctx_t *ctx)
{
    int rv;
    int data_len;
    char *pkt;
    jdwp_cmd_t header;
    jdwp_reply_t *reply = (jdwp_reply_t*)&header;

    rv = adb_read(ctx, (char*)&header, sizeof(jdwp_cmd_t));
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
	rv = adb_read(ctx, pkt+sizeof(header), data_len);
	if (rv < 0) {
	    free(pkt);
	    return NULL;
	}
    }

    return pkt;
}

int
jdwp_get_vm_version(adb_ctx_t *ctx)
{
    int rv;
    jdwp_reply_t *reply;

    jdwp_send_pkt(ctx, VIRTUALMACHINE_VERSION, NULL, 0);

    reply = (jdwp_reply_t*)jdwp_recv_pkt(ctx);
    if (reply == NULL) {
	return -1;
    }

    printf("%s\n", reply->data+4);
    free(reply);
    return 0;
}

int
jdwp_set_event_request(adb_ctx_t *ctx, char eventKind, char suspendPolicy)
{
    int requestID;
    jdwp_reply_t *reply;

    char data[] = {eventKind, suspendPolicy, 0, 0, 0, 0};

    jdwp_send_pkt(ctx, EVENTREQUEST_SET, data, sizeof(data));

    reply = (jdwp_reply_t*)jdwp_recv_pkt(ctx);
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
jdwp_resume_thread(adb_ctx_t *ctx, long long*threadID)
{
    jdwp_reply_t *reply;

    jdwp_send_pkt(ctx, THREADREF_RESUME, (char*)threadID, sizeof(*threadID));

    reply = (jdwp_reply_t*)jdwp_recv_pkt(ctx);
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

int
jdwp_read_events(adb_ctx_t *ctx)
{
    int rv;
    jdwp_cmd_t *cmd;

    int  i, offset;
    char *data;
    char suspendPolicy;
    int  events;
    char eventKind;

    while (1) {
	cmd  = (jdwp_cmd_t*)jdwp_recv_pkt(ctx);
	data = cmd->data;
	if (cmd == NULL) {
	    fprintf(stderr, "Error: JDWP: disconnect...\n");
	    break;
	}

	if (cmd->flags == 0x80) {
	    fprintf(stderr, "Error: JDWP: ignore a reply pkt.\n");
	    continue;
	}

	if (((cmd->cmd_set << 8) | cmd->command) != EVENT_COMPOSIT) {
	    fprintf(stderr, "Error: JDWP: ignore a command pkt (%d, %d)\n",
		    cmd->cmd_set, cmd->command);
	    continue;
	}

	suspendPolicy = data[0];
	events = ntohl(*(int*)(data+1));

	offset = 5;
	for (i=0; i<events; i++) {
	    eventKind = data[offset++];
	    switch (eventKind) {
	    case E_THREAD_START:
		printf("Get Event THREAD_START\n");
		if (suspendPolicy != 0) {
		    printf("Resume the thread\n");
		    jdwp_resume_thread(ctx, (long long*)(data+offset+4));
		}
		offset += 4 + 8; // requestID + thread
		break;
	    case E_THREAD_END:
		printf("Get Event THREAD_END\n");
		offset += 4 + 8; // requestID + thread
		break;
	    case E_VM_START:
		printf("Get Event VM_START\n");
		offset += 4 + 8; // requestID + thread
		break;
	    case E_VM_DEATH:
		printf("Get Event VM_DEATH\n");
		offset += 4; // requestID
		break;
	    default:
		fprintf(stderr, "Error: JDWP: ignore event %d\n", eventKind);
		break;
	    }
	}
    }

    return 0;
}

