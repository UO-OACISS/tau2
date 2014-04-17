#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>

#include "jdwp.h"
#include "ddm.h"

/*
 * Convert a 4-character string to a 32-bit type.
 * See ChunkHandler.java in Android source tree.
 */
int
ddm_trunk_type(char *name)
{
    int i;
    int type = 0;

    for (i=0; i<4; i++) {
	type = (type << 8) | name[i];
    }

    return type;
}

/*
int
ddm_send_trunk(jdwp_ctx_t *jdwp, uint32_t type, uint32_t len, char *data)
{
    int rv;
    ddm_trunk_t *trunk;
    int trunk_len = sizeof(type) + sizeof(len) + len;

    trunk = (ddm_trunk_t *)malloc(trunk_len);

    trunk->type   = htonl(type);
    trunk->length = htonl(len);
    memcpy(trunk->data, data, len);
    
    rv = jdwp_send_pkt(jdwp, DDM_TRUNK, trunk, trunk_len);
    return rv;
}

ddm_trunk_t *
ddm_get_reply(jdwp_ctx_t *jdwp)
{
    jdwp_reply_t *reply;

    reply = jdwp_get_reply(jdwp);
    if (reply == NULL) {
	return NULL;
    }

    return NULL;
}
*/

/* DDM initial handshake */
int
ddm_helo(jdwp_ctx_t *jdwp)
{
    char buf[8+4];
    jdwp_reply_t *reply;
    ddm_trunk_t *trunk = (ddm_trunk_t*)buf;

    trunk->type   = htonl(DDM_HELO);
    trunk->length = htonl(4);
    *(uint32_t*)(trunk->data) = htonl(1);

    jdwp_send_pkt(jdwp, DDM_TRUNK, (char*)trunk, sizeof(buf));

    reply = jdwp_get_reply(jdwp);
    if (reply == NULL) {
	return -1;
    }

    /*
     * we simply ignore the contents of HELO reply, just check the error code
     */
    if (reply->error_code != 0) {
	free(reply);
	return -1;
    }

    free(reply);
    return 0;
}

/* enable thread creation/death notification */
int
ddm_then(jdwp_ctx_t *jdwp)
{
    jdwp_reply_t *reply;
    ddm_then_t trunk;

    trunk.type   = htonl(DDM_THEN);
    trunk.length = htonl(1);
    trunk.enable = 1;

    jdwp_send_pkt(jdwp, DDM_TRUNK, (char*)&trunk, sizeof(trunk));

    reply = jdwp_get_reply(jdwp);
    if (reply == NULL) {
	return -1;
    }

    if (reply->error_code != 0) {
	free(reply);
	return -1;
    }

    free(reply);
    return 0;
}

/* get thread status */
ddm_thst_t *
ddm_thst(jdwp_ctx_t *jdwp)
{
    jdwp_reply_t *reply;
    ddm_trunk_t trunk;
    ddm_thst_t *thst;

    trunk.type   = htonl(DDM_THST);
    trunk.length = htonl(0);

    jdwp_send_pkt(jdwp, DDM_TRUNK, (char*)&trunk, sizeof(trunk));

    reply = jdwp_get_reply(jdwp);
    if (reply == NULL) {
	return NULL;
    }

    if (reply->error_code != 0) {
	free(reply);
	return NULL;
    }

    /* can we do zero copy? */
    thst = (ddm_thst_t*)malloc(reply->length - sizeof(*reply));
    memcpy(thst, reply->data, reply->length - sizeof(*reply));

    free(reply);
    return thst;
}
