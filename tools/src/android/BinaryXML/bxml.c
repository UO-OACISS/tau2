#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "bxml.h"

#define _V(...) do {					\
    int _i;						\
    for (_i=0; _i<level; _i++) {			\
	printf("    ");					\
    }							\
    printf(__VA_ARGS__);				\
    } while(0);

#define V(...) printf(__VA_ARGS__)

#define STR(i) (xml->stringPool->cstrings[i])

int bxml_open_node(bxml_t *xml, node_t *parent);
int bxml_close_node(bxml_t *xml, node_t *node);
int bxml_close_namespace(bxml_t *xml);
int bxml_binary_dump_node(bxml_t *xml, node_t *node);

/*
 * The binary format XML used in Android APK is not well documented. The only
 * way to understand the format is to read the header file in Android source
 * tree, i.e.
 *   <Android>/frameworks/base/include/androidfw/ResourceTypes.h
 */

/* convert an UTF-16LE string to a C string */
char *
utf16le_to_cstr(uint8_t *str)
{
    int i;
    uint16_t *utf;
    uint16_t len;
    char *cstr;

    utf = (uint16_t*)str;
    len = le16toh(utf[0]);

    cstr = (char*)malloc(len + 1);

    for (i=0; i<len+1; i++) {
	cstr[i] = (char)le16toh(utf[i+1]);
    }

    return cstr;
}

/* convert a C string to an UTF-16LE string */
uint8_t *
cstr_to_utf16le(char *str)
{
    int i;
    uint16_t *utf;
    uint16_t len = strlen(str);

    utf = (uint16_t*)malloc(sizeof(uint16_t) * (len + 2));

    utf[0] = htole16(len);

    for (i=0; i<len+1; i++) {
	utf[i+1] = htole16(str[i]);
    }

    return (uint8_t*)utf;
}

/*
 * String Pool
 * -----------------
 * (u2)  chunkType                  : RES_STRING_POOL_TYPE
 * (u2)  headerSize                 : 0x001c (chunkType ~ styleOffset)
 * (u4)  trunkSize                  : everything in this trunk
 * (u4)  stringCount                : entry count in stringIdxTable
 * (u4)  styleCount                 : entry count in styleIdxTable
 * (u4)  flags                      : UTF-8 / UTF-16LE
 * (u4)  stringOffset               : base address of strings
 * (u4)  styleOffset                : base address of styles
 * (u4*) stringIdxTable             : offset table of each string
 * (u4*) styleIdxTable              : offset table of each style
 * (*)   data = Strings + Styles    : strings and styles
 *
 * String (UTF-16LE)
 * -----------------
 * (u2)  stringLen
 * (u2*) stringData
 * (u2)  0x0000
 *
 * Style
 * -----------------
 * Don't care. We won't modify it.
 */
int
bxml_parse_strings(bxml_t *xml)
{
    int i, j;
    uint32_t offset, pos;
    uint32_t trunkBase;
    uint32_t stringBase;
    uint32_t styleBase;
    stringPool_t *stringPool;

    stringPool = (stringPool_t*)malloc(sizeof(*stringPool));

    trunkBase = brOffset(xml) - 2;

    stringPool->trunkType    = RES_STRING_POOL_TYPE;
    stringPool->headerSize   = brLE16(xml);
    stringPool->trunkSize    = brLE32(xml);

    stringPool->stringCount  = brLE32(xml);
    stringPool->styleCount   = brLE32(xml);
    stringPool->flags        = brLE32(xml);
    stringPool->stringOffset = brLE32(xml);
    stringPool->styleOffset  = brLE32(xml);

    stringPool->strings = (uint8_t**)malloc(sizeof(uint8_t*) *
					    stringPool->stringCount);
    stringPool->styles  = (style_t**)malloc(sizeof(style_t*) *
					    stringPool->styleCount);

    /* C string to make our life easier */
    stringPool->cstrings = (char **)malloc(sizeof(char*) * stringPool->stringCount);

    stringBase = trunkBase + stringPool->stringOffset;
    styleBase  = trunkBase + stringPool->styleOffset;

    /* read string table */
    for (i=0; i<stringPool->stringCount; i++) {
	/* get offset of this string */
	offset = brLE32(xml);

	/* save current position */
	pos    = brOffset(xml);

	/* move to the position of the string */
	brMoveTo(xml, stringBase + offset);

	/* get the string */
	if (stringPool->flags & UTF8_FLAG) { // UTF-8
	    fprintf(stderr, "Don't support UTF-8 yet... Abort.\n");
	    return -1;
	} else {                 // UTF-16LE
	    uint16_t len = brLE16(xml);

	    /* (u2)len + utf16_string + terminal_(u2)0x0000 */
	    len = sizeof(len) + 2 * len + 2;

	    stringPool->strings[i] = (uint8_t*)malloc(len);

	    brSkip(xml, -2);
	    for (j=0; j<len; j++) {
		stringPool->strings[i][j] = brByte(xml);
	    }

	    stringPool->cstrings[i] = utf16le_to_cstr(stringPool->strings[i]);

	    //printf("%s\n", utf16le_to_cstr(stringPool->strings[i]));
	}

	/* move back */
	brMoveTo(xml, pos);
    }

    /* read style table */
    for (i=0; i<stringPool->styleCount; i++) {
	/* get offset of this style_t */
	offset = brLE32(xml);

	/* save current position */
	pos    = brOffset(xml);

	/* move to the position of the style_t */
	brMoveTo(xml, styleBase + offset);

	/* get the style_t */
	stringPool->styles[i] = (style_t*)malloc(sizeof(style_t));
	stringPool->styles[i]->name      = brLE32(xml);
	stringPool->styles[i]->firstChar = brLE32(xml);
	stringPool->styles[i]->lastChar  = brLE32(xml);

	/* move back */
	brMoveTo(xml, pos);
    }

    /* we are done, now move to next trunk */
    brMoveTo(xml, trunkBase + stringPool->trunkSize);

    xml->stringPool = stringPool;

    return 0;
}

int
bxml_parse_resource_map(bxml_t *xml)
{
    int i;
    uint32_t trunkBase;
    resourceMap_t *map;

    trunkBase = brOffset(xml) - 2;

    uint16_t headerSize = brLE16(xml);
    uint32_t trunkSize  = brLE32(xml);

    uint32_t count = (trunkSize - headerSize) / sizeof(uint32_t);
    
    map = (resourceMap_t*)malloc(sizeof(*map) + (trunkSize - headerSize));

    map->trunkType  = RES_XML_RESOURCE_MAP_TYPE;
    map->headerSize = headerSize;
    map->trunkSize  = trunkSize;
    map->count      = count;

    for (i=0; i<count; i++) {
	map->ids[i] = brLE32(xml);
    }

    brMoveTo(xml, trunkBase + trunkSize);

    xml->resourceMap = map;
    
    return 0;
}

int
bxml_open_namespace(bxml_t *xml)
{
    uint32_t trunkBase;

    trunkBase = brOffset(xml) - 2;
    
    xml->ns.trunkType  = RES_XML_START_NAMESPACE_TYPE;
    xml->ns.headerSize = brLE16(xml);
    xml->ns.trunkSize  = brLE32(xml);

    xml->ns.lineNum    = brLE32(xml);
    xml->ns.comment    = brLE32(xml);
    xml->ns.prefix     = brLE32(xml);
    xml->ns.uri        = brLE32(xml);

    brMoveTo(xml, trunkBase + xml->ns.trunkSize);

    /* looking for RES_XML_END_NAMESPACE_TYPE */
    int closed = 0;
    while (!closed) {
	uint16_t trunkType;

	if (brEOF(xml)) {
	    printf("Error: Malformed XML. Namespace is not closed.\n");
	    return -1;
	}

	trunkType = brLE16(xml);

	switch (trunkType) {
	case RES_XML_START_ELEMENT_TYPE:
	    bxml_open_node(xml, NULL);
	    break;

	case RES_XML_END_NAMESPACE_TYPE:
	    bxml_close_namespace(xml);
	    closed = 1;
	    break;

	default:
	    printf("Error: Malformed XML. Need END_NS, get %04x.\n", trunkType);
	    return -1;
	}
    }

    return 0;
}

int
bxml_close_namespace(bxml_t *xml)
{
    uint32_t trunkBase  = brOffset(xml) - 2;

    xml->ns.end.headerSize = brLE16(xml);
    xml->ns.end.trunkSize  = brLE32(xml);
    
    xml->ns.end.lineNum = brLE32(xml);
    xml->ns.end.comment = brLE32(xml);
    xml->ns.end.prefix  = brLE32(xml);
    xml->ns.end.uri     = brLE32(xml);

    brMoveTo(xml, trunkBase + xml->ns.end.trunkSize);

    return 0;
}

int
bxml_open_node(bxml_t *xml, node_t *parent)
{
    int i;
    uint32_t trunkBase;
    node_t  *node;
    node_t  *sibling;

    trunkBase = brOffset(xml) - 2;

    node = (node_t*)malloc(sizeof(*node));

    node->trunkType  = RES_XML_START_ELEMENT_TYPE;
    node->headerSize = brLE16(xml);
    node->trunkSize  = brLE32(xml);

    node->lineNum    = brLE32(xml);
    node->comment    = brLE32(xml);

    node->ns         = brLE32(xml);
    node->name       = brLE32(xml);
    node->attrStart  = brLE16(xml);
    node->attrSize   = brLE16(xml); // equal to sizoef(attr_t) * attrCount
    node->attrCount  = brLE16(xml);
    node->idIndex    = brLE16(xml);
    node->classIndex = brLE16(xml);
    node->styleIndex = brLE16(xml);

    node->attrs = (attr_t*)malloc(sizeof(attr_t) * node->attrCount);

    for (i=0; i<node->attrCount; i++) {
	node->attrs[i].ns                  = brLE32(xml);
	node->attrs[i].name                = brLE32(xml);
	node->attrs[i].rawValue            = brLE32(xml);
	node->attrs[i].typedValue.size     = brLE16(xml);
	node->attrs[i].typedValue.res0     = brByte(xml);
	node->attrs[i].typedValue.dataType = brByte(xml);
	node->attrs[i].typedValue.data     = brLE32(xml);
    }

    node->child   = NULL;
    node->sibling = NULL;

    if (parent == NULL) {
	node->longname = strdup(STR(node->name));
    } else {
	node->longname = malloc(strlen(parent->longname) + 1 +
				strlen(STR(node->name))  + 1);
	node->longname[0] = 0;
	strcat(node->longname, parent->longname);
	strcat(node->longname, ".");
	strcat(node->longname, STR(node->name));
    }

    /* we are done with this trunk, move to next one */
    brMoveTo(xml, trunkBase + node->trunkSize);

    /* looking for RES_XML_END_ELEMENT_TYPE trunk */
    int closed = 0;
    while (!closed) {
	uint16_t trunkType;

	if (brEOF(xml)) {
	    printf("Error: Malformed XML. Node is not closed.\n");
	    return -1;
	}

	trunkType = brLE16(xml);

	switch (trunkType) {
	case RES_XML_START_ELEMENT_TYPE:
	    bxml_open_node(xml, node);
	    break;

	case RES_XML_END_ELEMENT_TYPE:
	    bxml_close_node(xml, node);
	    closed = 1;
	    break;

	default:
	    printf("Error: Malformed XML. Need END_NODE, get %04x.\n", trunkType);
	    return -1;
	}
    }

    /* hook the node */
    if (parent == NULL) {
	if (xml->node == NULL) {
	    xml->node = node;
	} else {
	    printf("Error: multiple toplevel node.\n");
	    return -1;
	}
    } else {
	if (parent->child == NULL) {
	    parent->child = node;
	} else {
	    sibling = parent->child;
	    while (sibling->sibling != NULL) {
		sibling = sibling->sibling;
	    }
	    sibling->sibling = node;
	}
    }
    
    return 0;
}

int
bxml_close_node(bxml_t *xml, node_t *node)
{
    uint32_t trunkBase  = brOffset(xml) - 2;

    node->end.headerSize = brLE16(xml);
    node->end.trunkSize  = brLE32(xml);

    node->end.lineNum    = brLE32(xml);
    node->end.comment    = brLE32(xml);
    node->end.ns         = brLE32(xml);
    node->end.name       = brLE32(xml);

    brMoveTo(xml, trunkBase + node->end.trunkSize);
    
    return 0;
}

/*
 * Trunk
 * ------------------
 * (u2) chunkType
 * (u2) headerSize
 * (u4) chunkSize
 * (*)  chunk_specific_data
 */
int
bxml_parse_trunks(bxml_t *xml)
{
    uint16_t trunkType;

    while (!brEOF(xml)) {
	trunkType = brLE16(xml);
	
	/*
	 * We handle only top level trunks here. Child trunks are
	 * handled recursivly.
	 */
	switch (trunkType) {
	case RES_STRING_POOL_TYPE:
	    bxml_parse_strings(xml);
	    break;

	case RES_XML_RESOURCE_MAP_TYPE:
	    bxml_parse_resource_map(xml);
	    break;

	case RES_XML_START_NAMESPACE_TYPE:
	    bxml_open_namespace(xml);
	    break;

	default:
	    printf("malformed xml: %04x\n", trunkType);
	    return -1;
	}
    }

    return 0;
}

/*
 * Andorid Binary XML
 * ------------------
 * (u4) CHUNK_AXML_FILE
 * (u4) file_size
 * (*)  chunks
 */
int
bxml_parse(bxml_t *xml)
{
    if (brLE32(xml) != CHUNK_AXML_FILE) {
	return -1;
    }

    if (brLE32(xml) != xml->size) {
	return -1;
    }

    return bxml_parse_trunks(xml);
}

bxml_t *
bxml_open(const char *filename)
{
    int rv;
    int fd;
    bxml_t *xml;
    struct stat st;

    xml = (bxml_t*)malloc(sizeof(*xml));
    bzero(xml, sizeof(*xml));

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
	perror("open");
	goto err_q_0;
    }

    rv = fstat(fd, &st);
    if (rv < 0) {
	perror("fstat");
	goto err_q_1;
    }

    xml->rdata = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (xml->rdata == MAP_FAILED) {
	perror("mmap");
	goto err_q_1;
    }

    xml->size = st.st_size;
    xml->rptr = xml->rdata;

    rv = bxml_parse(xml);
    
    /* close fd doesn't unmap the region */
    close(fd);

    return xml;

err_q_1:
    close(fd);
err_q_0:
    free(xml);
    return NULL;
}

void
bxml_free_node(node_t *node)
{
    node_t *n;

    while (node) {
	n = node;

	/* free the children ... */
	bxml_free_node(node->child);
	/* ... and going to free the sibling ... */
	node = node->sibling;
	/* ... before free myself */
	free(n->attrs);
	free(n->longname);
	free(n);
    }
}

int
bxml_close(bxml_t *xml)
{
    int i;

    munmap(xml->rdata, xml->size);

    /* free the memory */

    /* string pool */
    for (i=0; i<xml->stringPool->stringCount; i++) {
	free(xml->stringPool->strings[i]);
	free(xml->stringPool->cstrings[i]);
    }
    free(xml->stringPool->strings);
    free(xml->stringPool->cstrings);

    for (i=0; i<xml->stringPool->styleCount; i++) {
	free(xml->stringPool->styles[i]);
    }
    free(xml->stringPool->styles);

    free(xml->stringPool);

    /* resource map */
    free(xml->resourceMap);

    /* nodes */
    bxml_free_node(xml->node);

    /* xml */
    free(xml);

    return 0;
}

/*
 * NOTE: pos must be a signed type as ref may be -1.
 */
#define NEWREF(ref) ref = (ref<pos ? ref : ref+1)

static int
_node_update_string_ref(node_t *node, int pos)
{
    int i;

    while (node) {
	NEWREF(node->comment);
	NEWREF(node->ns);
	NEWREF(node->name);
	NEWREF(node->end.comment);
	NEWREF(node->end.ns);
	NEWREF(node->end.name);

	for (i=0; i<node->attrCount; i++) {
	    NEWREF(node->attrs[i].ns);
	    NEWREF(node->attrs[i].name);
	    NEWREF(node->attrs[i].rawValue);

	    if (node->attrs[i].typedValue.dataType == TYPE_STRING) {
		NEWREF(node->attrs[i].typedValue.data);
	    }
	}

	if (node->child != NULL) {
	    _node_update_string_ref(node->child, pos);
	}

	node = node->sibling;
    }

    return 0;
}

int
bxml_update_string_ref(bxml_t *xml, int pos)
{
    int i;

    for (i=0; i<xml->stringPool->styleCount; i++) {
	NEWREF(xml->stringPool->styles[i]->name);
    }

    NEWREF(xml->ns.comment);
    NEWREF(xml->ns.prefix);
    NEWREF(xml->ns.uri);
    NEWREF(xml->ns.end.comment);
    NEWREF(xml->ns.end.prefix);
    NEWREF(xml->ns.end.uri);

    _node_update_string_ref(xml->node, pos);

    return 0;
}

#undef NEWREF

/* search for the c string in string pool, return the index, or -1 */
int
bxml_find_cstring(bxml_t *xml, char *str)
{
    int i;

    for (i=0; i<xml->stringPool->stringCount; i++) {
	if (strcmp(str, STR(i)) == 0) {
	    return i;
	}
    }

    return -1;
}

/* add a c string into string pool, if it doesn't exist yet */
int
bxml_add_cstring_at(bxml_t *xml, char *str, int pos)
{
    int i;
    int index;
    uint8_t *utf;
    uint16_t len;
    stringPool_t *pool = xml->stringPool;

    uint8_t **strings  = pool->strings;
    char    **cstrings = pool->cstrings;

    if (pos > pool->stringCount) {
	return -1;
    }

    index = bxml_find_cstring(xml, str);
    if (index >= 0) {
	/* it's already in the pool */
	return index;
    }

    /* get a UTF-16LE string */
    utf = cstr_to_utf16le(str);
    len = le16toh( ((uint16_t*)utf)[0] );

    /* insert it into the string pool */

    pool->stringCount  += 1;
    pool->stringOffset += 4;
    if (pool->styleCount > 0) {
	pool->styleOffset  += 4 + sizeof(uint16_t) * (len + 2);
    }
    pool->strings = (uint8_t**)malloc(sizeof(uint8_t*) * pool->stringCount);
    pool->cstrings = (char **)malloc(sizeof(char*) * pool->stringCount);

    for (i=0; i<pos; i++) {
	pool->strings[i]  = strings[i];
	pool->cstrings[i] = cstrings[i];
    }

    // i == pos
    pool->strings[i]  = utf;
    pool->cstrings[i] = strdup(str);

    for (i+=1; i<pool->stringCount; i++) {
	pool->strings[i]  = strings[i-1];
	pool->cstrings[i] = cstrings[i-1];
    }

    free(strings);
    free(cstrings);

    pool->trunkSize += 4 + sizeof(uint16_t) * (len + 2);
    xml->size       += 4 + sizeof(uint16_t) * (len + 2);

    bxml_update_string_ref(xml, pos);

    return pos;
}

/* add a c string into the end of string pool, if it doesn't exist yet */
int
bxml_add_cstring(bxml_t *xml, char *str)
{
    stringPool_t *pool = xml->stringPool;

    return bxml_add_cstring_at(xml, str, pool->stringCount);
}

/*
 * find a node in xml
 * nodespec: name(.name)*
 */
node_t *
bxml_find_node(node_t *node, char *nodespec, node_t *last)
{
    node_t *n;
    int begin;

    if (last == NULL) {
	begin = 1;
    } else {
	begin = 0;
    }

    while (node) {
	if (strcmp(node->longname, nodespec) == 0) {
	    if (begin) {
		return node;
	    } else {
		if (node == last) {
		    begin = 1;
		}
	    }
	}

	n = bxml_find_node(node->child, nodespec, last);
	if (n != NULL) {
	    return n;
	} else {
	    node = node->sibling;
	}
    }

    return NULL;
}

int
bxml_text_dump_node(bxml_t *xml, node_t *node, int level)
{
    int i;
    attr_t *attr;

    while (node) {
	_V("<%s", STR(node->name));

	/* dump attributes */
	for (i=0; i<node->attrCount; i++) {
	    attr = &node->attrs[i];
	    switch (attr->typedValue.dataType) {
	    case TYPE_STRING:
		V(" %s:%s=\"%s\"", STR(xml->ns.prefix), STR(attr->name),
		  STR(attr->typedValue.data));
		break;

	    case TYPE_INT_BOOLEAN:
		V(" %s:%s=\"%s\"", STR(xml->ns.prefix), STR(attr->name),
		  attr->typedValue.data!=0 ? "true" : "false");
		break;

	    default:
		V(" %s:%s=%02x:%08x", STR(xml->ns.prefix), STR(attr->name),
		  attr->typedValue.dataType, attr->typedValue.data);
		break;
	    }
	}

	if (node->child == NULL) {
	    V("/>\n");
	} else {
	    V(">\n");
	    bxml_text_dump_node(xml, node->child, level+1);
	    _V("</%s>\n", STR(node->end.name));
	}

	node = node->sibling;
    }

    return 0;
}

int
bxml_text_dump(bxml_t *xml)
{
    /* namespace */
    V("<NS prefix=\"%s\" uri=\"%s\"/>\n", STR(xml->ns.prefix), STR(xml->ns.uri));

    /* node */
    bxml_text_dump_node(xml, xml->node, 0);

    return 0;
}

int
bxml_binary_dump_string_pool(bxml_t *xml)
{
    int i, j;
    stringPool_t *pool;
    uint32_t trunkBase;
    uint32_t stringBase;
    uint32_t styleBase;

    uint32_t offset, pos;

    pool = xml->stringPool;

    trunkBase = bwOffset(xml);

    /* write string pool headers */
    
    bwLE16(xml, RES_STRING_POOL_TYPE);
    bwLE16(xml, pool->headerSize);
    bwLE32(xml, pool->trunkSize);

    bwLE32(xml, pool->stringCount);
    bwLE32(xml, pool->styleCount);
    bwLE32(xml, pool->flags);
    bwLE32(xml, pool->stringOffset);
    bwLE32(xml, pool->styleOffset);

    stringBase    = trunkBase + pool->stringOffset;
    styleBase     = trunkBase + pool->styleOffset;

    /* write string table */
    offset = 0;
    for (i=0; i<pool->stringCount; i++) {
	uint16_t *v = (uint16_t*)pool->strings[i];
	uint16_t len = le16toh(*v);

	len = sizeof(len) + 2 * len + 2;

	bwLE32(xml, offset);

	pos = bwOffset(xml);

	bwMoveTo(xml, stringBase + offset);

	for (j=0; j<len; j++) {
	    bwByte(xml, pool->strings[i][j]);
	}
	offset += len;

	bwMoveTo(xml, pos);
    }

    /* write style table */
    offset = 0;
    for (i=0; i<pool->styleCount; i++) {
	bwLE32(xml, offset);

	pos = bwOffset(xml);

	bwMoveTo(xml, styleBase + offset);

	bwLE32(xml, pool->styles[i]->name);
	bwLE32(xml, pool->styles[i]->firstChar);
	bwLE32(xml, pool->styles[i]->lastChar);
	offset += 12;

	bwMoveTo(xml, pos);
    }

    bwMoveTo(xml, trunkBase + pool->trunkSize);
    
    return 0;
}

int
bxml_binary_dump_resource_map(bxml_t *xml)
{
    int i;
    resourceMap_t *map;

    map = xml->resourceMap;

    /* write resource map headers */
    bwLE16(xml, RES_XML_RESOURCE_MAP_TYPE);
    bwLE16(xml, map->headerSize);
    bwLE32(xml, map->trunkSize);

    /* write resource map */
    for (i=0; i<map->count; i++) {
	bwLE32(xml, map->ids[i]);
    }
    
    return 0;
}

int
bxml_binary_dump_start_namespace(bxml_t *xml)
{
    bwLE16(xml, RES_XML_START_NAMESPACE_TYPE);
    bwLE16(xml, xml->ns.headerSize);
    bwLE32(xml, xml->ns.trunkSize);

    bwLE32(xml, xml->ns.lineNum);
    bwLE32(xml, xml->ns.comment);
    bwLE32(xml, xml->ns.prefix);
    bwLE32(xml, xml->ns.uri);

    return 0;
}

int
bxml_binary_dump_end_namespace(bxml_t *xml)
{
    bwLE16(xml, RES_XML_END_NAMESPACE_TYPE);
    bwLE16(xml, xml->ns.end.headerSize);
    bwLE32(xml, xml->ns.end.trunkSize);

    bwLE32(xml, xml->ns.end.lineNum);
    bwLE32(xml, xml->ns.end.comment);
    bwLE32(xml, xml->ns.end.prefix);
    bwLE32(xml, xml->ns.end.uri);

    return 0;
}

int
bxml_binary_dump_namespace(bxml_t *xml)
{
    bxml_binary_dump_start_namespace(xml);

    bxml_binary_dump_node(xml, xml->node);

    bxml_binary_dump_end_namespace(xml);

    return 0;
}

int
bxml_binary_dump_start_node(bxml_t *xml, node_t *node)
{
    int i;

    bwLE16(xml, RES_XML_START_ELEMENT_TYPE);
    bwLE16(xml, node->headerSize);
    bwLE32(xml, node->trunkSize);

    bwLE32(xml, node->lineNum);
    bwLE32(xml, node->comment);

    bwLE32(xml, node->ns);
    bwLE32(xml, node->name);
    bwLE16(xml, node->attrStart);
    bwLE16(xml, node->attrSize);
    bwLE16(xml, node->attrCount);
    bwLE16(xml, node->idIndex);
    bwLE16(xml, node->classIndex);
    bwLE16(xml, node->styleIndex);

    for (i=0; i<node->attrCount; i++) {
	bwLE32(xml, node->attrs[i].ns);
	bwLE32(xml, node->attrs[i].name);
	bwLE32(xml, node->attrs[i].rawValue);
	bwLE16(xml, node->attrs[i].typedValue.size);
	bwByte(xml, node->attrs[i].typedValue.res0);
	bwByte(xml, node->attrs[i].typedValue.dataType);
	bwLE32(xml, node->attrs[i].typedValue.data);
    }
    
    return 0;
}

int
bxml_binary_dump_end_node(bxml_t *xml, node_t *node)
{
    bwLE16(xml, RES_XML_END_ELEMENT_TYPE);
    bwLE16(xml, node->end.headerSize);
    bwLE32(xml, node->end.trunkSize);

    bwLE32(xml, node->end.lineNum);
    bwLE32(xml, node->end.comment);
    bwLE32(xml, node->end.ns);
    bwLE32(xml, node->end.name);

    return 0;
}

int
bxml_binary_dump_node(bxml_t *xml, node_t *node)
{
    while (node) {
	bxml_binary_dump_start_node(xml, node);

	if (node->child != NULL) {
	    bxml_binary_dump_node(xml, node->child);
	}

	bxml_binary_dump_end_node(xml, node);

	node = node->sibling;
    }

    return 0;
}

int
bxml_binary_dump(bxml_t *xml, char *filename)
{
    int fd;
    int pad;

    /* make sure string pool is 4-bytes aligned */
    pad = (4 - (xml->stringPool->trunkSize & 0x03)) & 0x03;
    xml->stringPool->trunkSize += pad;
    xml->size                  += pad;

    fd = open(filename, O_RDWR | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
	perror("open");
	return -1;
    }

    ftruncate(fd, xml->size);

    xml->wdata = mmap(NULL, xml->size,
		      PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (xml->wdata == MAP_FAILED) {
	perror("mmap");
	close(fd);
	unlink(filename);
	return -1;
    }

    xml->wptr = xml->wdata;

    bwLE32(xml, CHUNK_AXML_FILE);
    bwLE32(xml, xml->size);

    bxml_binary_dump_string_pool(xml);
    bxml_binary_dump_resource_map(xml);
    bxml_binary_dump_namespace(xml);

    munmap(xml->wdata, xml->size);
    close(fd);

    return 0;
}

int
bxml_enable_debug(bxml_t *xml)
{
    int i;
    node_t *node;
    attr_t *attr, *attrs;
    resourceMap_t *map;
    int pos;

    node = bxml_find_node(xml->node, "manifest.application", NULL);
    if (node == NULL) {
	/* this should never happen */
	return -1;
    }

    /* do we already have "debuggable" attribute in manifast.application? */
    for (i=0; i<node->attrCount; i++) {
	attr = &node->attrs[i];

	if (attr->name < xml->resourceMap->count) {
	    if (xml->resourceMap->ids[attr->name] < 0x0101000f) {
		pos = i;
	    }
	}

	if (strcmp(STR(attr->name), "debuggable") == 0) {
	    /* yes, now just need to make sure the value is true */
	    attr->rawValue        = -1;
	    attr->typedValue.data = 0xffffffff;

	    /* we are done, nothing else to do */
	    return 0;
	}
    }

    /* no, we do not, let's add it */
    pos += 1;
    node->attrCount += 1;

    attrs = (attr_t*)malloc(sizeof(attr_t) * node->attrCount);

    /* looks like attributes must appear in order of their resource id */
    for (i=0; i<pos; i++) {
	attrs[i] = node->attrs[i];
    }
    attr = &attrs[pos];
    for (i=pos+1; i<node->attrCount; i++) {
	attrs[i] = node->attrs[i-1];
    }
    free(node->attrs);
    node->attrs = attrs;

    attr->ns                  =
	bxml_add_cstring(xml, "http://schemas.android.com/apk/res/android");
    attr->name                = bxml_add_cstring_at(xml, "debuggable", 0);
    attr->rawValue            = -1;
    attr->typedValue.size     = 8;
    attr->typedValue.res0     = 0;
    attr->typedValue.dataType = TYPE_INT_BOOLEAN;
    attr->typedValue.data     = 0xffffffff;

    node->trunkSize += sizeof(attr_t);
    xml->size       += sizeof(attr_t);

    /* update resource map */
    xml->resourceMap->count += 1;
    map = (resourceMap_t*)malloc(sizeof(*map) +
				 sizeof(uint32_t) * xml->resourceMap->count);
    map->trunkType = xml->resourceMap->trunkType;
    map->headerSize = xml->resourceMap->headerSize;
    map->trunkSize = xml->resourceMap->trunkSize + 4;
    map->count = xml->resourceMap->count;

    /*
     * Resource ID for attribute name "debuggable" is always 0x0101000f.
     * See <Android>/system/frameworks/base/core/res/res/values/public.xml
     */
    map->ids[0] = 0x0101000f;  // array index is 0 because attr->name==0
    for (i=1; i<map->count; i++) {
	map->ids[i] = xml->resourceMap->ids[i-1];
    }

    free(xml->resourceMap);
    xml->resourceMap = map;
    xml->size += 4;

    /* we are done */
    return 0;
}

int
bxml_enable_permission(bxml_t *xml, char *permission)
{
    int i;
    attr_t *attr;
    node_t *node = NULL;
    node_t *perm;

    /* do we already have the permissin set? */
    while (1) {
	node = bxml_find_node(xml->node, "manifest.uses-permission", node);
	if (node == NULL) {
	    break;
	}

	for (i=0; i<node->attrCount; i++) {
	    attr = &node->attrs[i];
	    if ((strcmp(STR(attr->name), "name") == 0) &&
		(strcmp(STR(attr->typedValue.data), permission) == 0)) {
		/* yes, we are good, just return */
		return 0;
	    }
	}
    }

    /* no, we do not, let's add it */
    perm = (node_t*)malloc(sizeof(node_t));

    perm->trunkType  = RES_XML_START_ELEMENT_TYPE;
    perm->headerSize = 16;
    perm->trunkSize  = 56;

    perm->lineNum    = 0;
    perm->comment    = -1;

    perm->ns         = -1;
    perm->name       = bxml_add_cstring(xml, "uses-permission");
    perm->attrStart  = 20;
    perm->attrSize   = 20;
    perm->attrCount  = 1;
    perm->idIndex    = 0;
    perm->classIndex = 0;
    perm->styleIndex = 0;

    perm->attrs = (attr_t*)malloc(sizeof(attr_t) * perm->attrCount);

    perm->attrs[0].ns                  =
	bxml_add_cstring(xml, "http://schemas.android.com/apk/res/android");
    perm->attrs[0].name                = bxml_add_cstring(xml, "name");
    perm->attrs[0].rawValue            =
	bxml_add_cstring(xml, permission);
    perm->attrs[0].typedValue.size     = 8;
    perm->attrs[0].typedValue.res0     = 0;
    perm->attrs[0].typedValue.dataType = TYPE_STRING;
    perm->attrs[0].typedValue.data     =
	bxml_add_cstring(xml, permission);

    perm->end.headerSize = 16;
    perm->end.trunkSize  = 24;
    perm->end.lineNum    = 0;
    perm->end.comment    = -1;
    perm->end.ns         = -1;
    perm->end.name       = bxml_add_cstring(xml, "uses-permission");

    perm->longname = strdup("manifest.uses-permission");

    perm->child = NULL;

    /* hook our new node after manifest.uses-sdk */
    node = bxml_find_node(xml->node, "manifest.uses-sdk", NULL);
    perm->sibling = node->sibling;
    node->sibling = perm;

    /* enlarge the size of xml */
    xml->size += perm->trunkSize + perm->end.trunkSize;

    /* we are done */
    return 0;
}

int
bxml_inject(bxml_t *xml)
{
    bxml_enable_debug(xml);
    bxml_enable_permission(xml, "android.permission.INTERNET");
    bxml_enable_permission(xml, "android.permission.WRITE_EXTERNAL_STORAGE");

    return 0;
}

const char *usage = 
"Usage: bxml [-d] [-i <NewManifest.xml>] <AndroidManifest.xml>\n"
"\n"
"Options:\n"
"    -d: Dump the text representation of <AndroidManifest.xml> to stdout. Dump\n"
"        injected XML if \"-i\" is given. (Default if no option given)\n"
"    -i: Inject <AndroidManifest.xml> to make sure android.permission.INTERNET\n"
"        is set and JDWP is enabled. Injected XML will be written into \n"
"        <NewManifest.xml>.\n";

int
main(int argc, char **argv)
{
    bxml_t *xml;

    int opt;
    int dump = 0;
    int inject = 0;
    char *newxml;

    opterr = 0;
    while ((opt = getopt(argc, argv, "di:")) != -1) {
	switch (opt) {
	case 'i':
	    inject = 1;
	    newxml = optarg;
	    break;
	case 'd':
	    dump = 1;
	    break;
	default:
	    fprintf(stderr, usage);
	    return -1;
	}
    }

    if (optind >= argc) {
	fprintf(stderr, usage);
	return -1;
    }

    xml = bxml_open(argv[optind]);

    if (inject) {
	bxml_inject(xml);
	bxml_binary_dump(xml, newxml);
	if (dump) {
	    bxml_text_dump(xml);
	}
    } else {
	bxml_text_dump(xml);
    }

    bxml_close(xml);

    return 0;
}
