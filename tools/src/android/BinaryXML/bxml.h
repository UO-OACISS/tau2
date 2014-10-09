#ifndef _BXML_H_
#define _BXML_H_

#include <stdint.h>
#define _BSD_SOURCE
#ifdef __APPLE_CC__
#include <machine/endian.h>
// Warning: assuming little endian host (x86_64, i386, etc.)
#ifndef le16toh
#define le16toh(x) (x)
#endif
#ifndef htole16
#define htole16(x) (x)
#endif
#ifndef le32toh
#define le32toh(x) (x)
#endif
#ifndef htole32
#define htole32(x) (x)
#endif
#else
#include <endian.h>
#endif
#include <sys/types.h>

#define CHUNK_AXML_FILE           0x00080003
#define CHUNK_RESOURCEIDS         0x00080180
#define CHUNK_STRINGS             0x001C0001
#define CHUNK_XML_END_NAMESPACE   0x00100101
#define CHUNK_XML_END_TAG         0x00100103
#define CHUNK_XML_START_NAMESPACE 0x00100100
#define CHUNK_XML_START_TAG       0x00100102
#define CHUNK_XML_TEXT            0x00100104

enum {
    RES_NULL_TYPE               = 0x0000,
    RES_STRING_POOL_TYPE        = 0x0001,
    RES_TABLE_TYPE              = 0x0002,
    RES_XML_TYPE                = 0x0003,

    // Chunk types in RES_XML_TYPE
    RES_XML_FIRST_CHUNK_TYPE    = 0x0100,
    RES_XML_START_NAMESPACE_TYPE= 0x0100,
    RES_XML_END_NAMESPACE_TYPE  = 0x0101,
    RES_XML_START_ELEMENT_TYPE  = 0x0102,
    RES_XML_END_ELEMENT_TYPE    = 0x0103,
    RES_XML_CDATA_TYPE          = 0x0104,
    RES_XML_LAST_CHUNK_TYPE     = 0x017f,
    // This contains a uint32_t array mapping strings in the string
    // pool back to resource identifiers.  It is optional.
    RES_XML_RESOURCE_MAP_TYPE   = 0x0180,

    // Chunk types in RES_TABLE_TYPE
    RES_TABLE_PACKAGE_TYPE      = 0x0200,
    RES_TABLE_TYPE_TYPE         = 0x0201,
    RES_TABLE_TYPE_SPEC_TYPE    = 0x0202
};

#define UTF8_FLAG  0x00000100

typedef struct {
    uint16_t trunk_type;
    uint16_t header_size;
    uint32_t trunk_size;
} trunk_header_t;


// Index to the string table in string pool
typedef int stringRef_t;

typedef struct {
    stringRef_t name;
    uint32_t    firstChar;
    uint32_t    lastChar;
} style_t;

typedef struct {
    uint16_t trunkType;
    uint16_t headerSize;
    uint32_t trunkSize;

    uint32_t stringCount;
    uint32_t styleCount;
    uint32_t flags;
    uint32_t stringOffset;
    uint32_t styleOffset;

    uint8_t **strings;
    style_t **styles;

    char **cstrings;
} stringPool_t;

typedef struct {
    uint16_t trunkType;
    uint16_t headerSize;
    uint32_t trunkSize;

    uint32_t count;
    uint32_t ids[];
} resourceMap_t;

typedef struct {
    uint16_t trunkType;
    uint16_t headerSize;
    uint32_t trunkSize;

    uint32_t    lineNum;
    stringRef_t comment;

    stringRef_t prefix;
    stringRef_t uri;

    struct {
	uint16_t    headerSize;
	uint32_t    trunkSize;

	uint32_t    lineNum;
	stringRef_t comment;
	stringRef_t prefix;
	stringRef_t uri;
    } end;
} namespace_t;

/* resValue_t.dataType */
enum {
    // Contains no data.
    TYPE_NULL = 0x00,
    // The 'data' holds a ResTable_ref, a reference to another resource
    // table entry.
    TYPE_REFERENCE = 0x01,
    // The 'data' holds an attribute resource identifier.
    TYPE_ATTRIBUTE = 0x02,
    // The 'data' holds an index into the containing resource table's
    // global value string pool.
    TYPE_STRING = 0x03,
    // The 'data' holds a single-precision floating point number.
    TYPE_FLOAT = 0x04,
    // The 'data' holds a complex number encoding a dimension value,
    // such as "100in".
    TYPE_DIMENSION = 0x05,
    // The 'data' holds a complex number encoding a fraction of a
    // container.
    TYPE_FRACTION = 0x06,

    // Beginning of integer flavors...
    TYPE_FIRST_INT = 0x10,

    // The 'data' is a raw integer value of the form n..n.
    TYPE_INT_DEC = 0x10,
    // The 'data' is a raw integer value of the form 0xn..n.
    TYPE_INT_HEX = 0x11,
    // The 'data' is either 0 or 1, for input "false" or "true" respectively.
    TYPE_INT_BOOLEAN = 0x12,

    // Beginning of color integer flavors...
    TYPE_FIRST_COLOR_INT = 0x1c,

    // The 'data' is a raw integer value of the form #aarrggbb.
    TYPE_INT_COLOR_ARGB8 = 0x1c,
    // The 'data' is a raw integer value of the form #rrggbb.
    TYPE_INT_COLOR_RGB8 = 0x1d,
    // The 'data' is a raw integer value of the form #argb.
    TYPE_INT_COLOR_ARGB4 = 0x1e,
    // The 'data' is a raw integer value of the form #rgb.
    TYPE_INT_COLOR_RGB4 = 0x1f,

    // ...end of integer flavors.
    TYPE_LAST_COLOR_INT = 0x1f,

    // ...end of integer flavors.
    TYPE_LAST_INT = 0x1f
};

typedef struct {
    uint16_t size;
    uint8_t  res0; // always set to 0
    uint8_t  dataType;
    uint32_t data;
} resValue_t;

typedef struct {
    stringRef_t ns;
    stringRef_t name;
    stringRef_t rawValue;
    resValue_t  typedValue;
} attr_t;

typedef struct _node {
    uint16_t trunkType;
    uint16_t headerSize;
    uint32_t trunkSize;

    uint32_t    lineNum;
    stringRef_t comment;

    stringRef_t ns;
    stringRef_t name;
    uint16_t    attrStart;
    uint16_t    attrSize;
    uint16_t    attrCount;
    uint16_t    idIndex;
    uint16_t    classIndex;
    uint16_t    styleIndex;

    struct {
	uint16_t    headerSize;
	uint32_t    trunkSize;

	uint32_t    lineNum;
	stringRef_t comment;
	stringRef_t ns;
	stringRef_t name;
    } end;

    attr_t       *attrs;

    struct _node *child;
    struct _node *sibling;
    char         *longname;
} node_t;

typedef struct {
    uint8_t *rdata;
    uint8_t *wdata;

    off_t    size;

    uint8_t *rptr;
    uint8_t *wptr;

    namespace_t    ns;
    stringPool_t  *stringPool;
    resourceMap_t *resourceMap;
    node_t        *node;
} bxml_t;

static inline uint8_t
brByte(bxml_t *xml)
{
    return *(xml->rptr)++;
}

static inline uint16_t
brLE16(bxml_t *xml)
{
    uint16_t *d = (uint16_t*)xml->rptr;

    xml->rptr += sizeof(*d);

    return le16toh(*d);
}

static inline uint32_t
brLE32(bxml_t *xml)
{
    uint32_t *d = (uint32_t*)xml->rptr;

    xml->rptr += sizeof(*d);

    return le32toh(*d);
}

static inline long
brLeb128(bxml_t *xml)
{
    int  i = 0;
    long v = 0;
    uint8_t b;

    while (1) {
	b = brByte(xml);

	v |= ((long)(b & 0x7f)) << i;

	i += 7;

	if (b & 0x80) {
	    break;
	}
    }

    if (v & (1L << (i-1))) {
	v -= 1L << i;
    }

    return v;
}

static inline void
brMoveTo(bxml_t *xml, int offset)
{
    xml->rptr = xml->rdata + offset;
}

static inline int
brOffset(bxml_t *xml)
{
    return xml->rptr - xml->rdata;
}

static inline void
brSkip(bxml_t *xml, int bytes)
{
    xml->rptr += bytes;
}

static inline int
brEOF(bxml_t *xml)
{
    return (xml->rptr >= xml->rdata + xml->size);
}

static inline void
bwByte(bxml_t *xml, uint8_t v)
{
    uint8_t *d = (uint8_t*)xml->wptr;

    *d = v;

    xml->wptr += sizeof(*d);
}

static inline void
bwLE16(bxml_t *xml, uint16_t v)
{
    uint16_t *d = (uint16_t*)xml->wptr;

    *d = htole16(v);

    xml->wptr += sizeof(*d);
}

static inline void
bwLE32(bxml_t *xml, uint32_t v)
{
    uint32_t *d = (uint32_t*)xml->wptr;

    *d = htole32(v);

    xml->wptr += sizeof(*d);
}

static inline void
bwMoveTo(bxml_t *xml, int offset)
{
    xml->wptr = xml->wdata + offset;
}

static inline int
bwOffset(bxml_t *xml)
{
    return xml->wptr - xml->wdata;
}

static inline void
bwSkip(bxml_t *xml, int bytes)
{
    xml->wptr += bytes;
}

#endif
