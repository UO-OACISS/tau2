/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauSampling.cpp                                  **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://tau.uoregon.edu                       **
**                                                                         **
**      Description     : This file contains all the XML related code      **
**                                                                         **
****************************************************************************/


#include <TauUtil.h>
#include <TauMetrics.h>
#include <TauMetaData.h>
#include <TauXML.h>

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <string>

/*********************************************************************
 * writes an XML string to an output device, converts certain
 * characters as necessary and uses CDATA when necessary
 ********************************************************************/
void Tau_XML_writeString(Tau_util_outputDevice *out, const char *s) {
  if (!s) return;

  bool useCdata = false;

  if (strchr(s, '<') || strchr(s, '&')) {
    useCdata = true;
  }

  if (strstr(s, "]]>") || strchr(s, '\n')) {
    useCdata = false;
  }

  if (useCdata) {
    Tau_util_output (out,"<![CDATA[%s]]>",s);
    return;
  }

  // could grow up to 5 times in length
  std::string buffer;
  std::string data(s);
  //buffer.reserve(data.size()*6);
    for(size_t pos = 0; pos != data.size(); ++pos) {
        switch(data[pos]) {
            case '&':  buffer.append("&amp;");       break;
            case '\"': buffer.append("&quot;");      break;
            case '\'': buffer.append("&apos;");      break;
            case '\n': buffer.append("&#xa;");      break;
            case '<':  buffer.append("&lt;");        break;
            case '>':  buffer.append("&gt;");        break;
            default:   buffer.append(&data[pos], 1); break;
        }
    }

  Tau_util_output (out,"%s",buffer.c_str());
}

/*********************************************************************
 * writes an XML tag
 ********************************************************************/
void Tau_XML_writeTag(Tau_util_outputDevice *out, const char *tag, const char *str, bool newline) {
  Tau_util_output (out, "<%s>", tag);
  Tau_XML_writeString(out, str);
  Tau_util_output (out, "</%s>",tag);
  if (newline) {
    Tau_util_output (out, "\n");
  }
}


/*********************************************************************
 * writes an attribute entity with a string value
 ********************************************************************/
void Tau_XML_writeAttribute(Tau_util_outputDevice *out, const char *name, const char *value, bool newline) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  Tau_util_output (out, "<attribute>%s<name>", endl);
  Tau_XML_writeString(out, name);
  Tau_util_output (out, "</name>%s<value>", endl);
  Tau_XML_writeString(out, value);
  Tau_util_output (out, "</value>%s</attribute>%s", endl, endl);
}

void Tau_XML_writeAttribute(Tau_util_outputDevice *out, const Tau_metadata_array_t *array, bool newline) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  int i;
  for (i = 0 ; i < array->length ; i++) {
    Tau_util_output (out, "<array_element>", endl);
	Tau_metadata_value_t *metadata = array->values[i];
    switch (metadata->type) {
      case TAU_METADATA_TYPE_STRING:
        Tau_XML_writeString(out, metadata->data.cval);
	    break;
      case TAU_METADATA_TYPE_INTEGER:
        Tau_util_output (out,"%d",metadata->data.ival);
	    break;
      case TAU_METADATA_TYPE_DOUBLE:
        Tau_util_output (out,"%f",metadata->data.dval);
	    break;
      case TAU_METADATA_TYPE_NULL:
        Tau_util_output (out,"NULL");
	    break;
      case TAU_METADATA_TYPE_FALSE:
        Tau_util_output (out,"FALSE");
	    break;
      case TAU_METADATA_TYPE_TRUE:
        Tau_util_output (out,"TRUE");
	    break;
      case TAU_METADATA_TYPE_ARRAY:
        Tau_XML_writeAttribute(out, metadata->data.aval, newline);
	    break;
      case TAU_METADATA_TYPE_OBJECT:
	    for (int i = 0 ; i < metadata->data.oval->count; i++) {
	      Tau_metadata_key *key = new Tau_metadata_key();
	      key->name = strdup(metadata->data.oval->names[i]);
          Tau_XML_writeAttribute(out, key, metadata->data.oval->values[i], newline);
		}
	    break;
    }
    Tau_util_output (out, "</array_element>", endl);
  }
}


/*********************************************************************
 * writes a complex attribute value
 ********************************************************************/
void Tau_XML_writeAttribute(Tau_util_outputDevice *out, const Tau_metadata_key *key, const Tau_metadata_value_t *metadata, bool newline) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  //Tau_util_output (out, "<attribute timer=\"%s\" call_number=%d timestamp=%llu>%s<name>", key->timer_context, key->call_number, key->timestamp, endl);
  Tau_util_output (out, "<attribute>%s<name>", endl);
  Tau_XML_writeString(out, key->name);
  if (key->timer_context == NULL) {
    Tau_util_output (out, "</name>%s<value>", endl);
  } else {
    Tau_util_output (out, "</name>%s<timer_context>", endl);
    Tau_XML_writeString(out, key->timer_context);
    Tau_util_output (out, "</timer_context>%s<call_number>", endl);
    Tau_util_output (out, "%d", key->call_number);
    Tau_util_output (out, "</call_number>%s<timestamp>", endl);
    Tau_util_output (out, "%llu", key->timestamp);
    Tau_util_output (out, "</timestamp>%s<value>", endl);
  }
  switch (metadata->type) {
    case TAU_METADATA_TYPE_STRING:
      Tau_XML_writeString(out, metadata->data.cval);
	  break;
    case TAU_METADATA_TYPE_INTEGER:
      Tau_util_output (out,"%d",metadata->data.ival);
	  break;
    case TAU_METADATA_TYPE_DOUBLE:
      Tau_util_output (out,"%f",metadata->data.dval);
	  break;
    case TAU_METADATA_TYPE_NULL:
      Tau_util_output (out,"NULL");
	  break;
    case TAU_METADATA_TYPE_FALSE:
      Tau_util_output (out,"FALSE");
	  break;
    case TAU_METADATA_TYPE_TRUE:
      Tau_util_output (out,"TRUE");
	  break;
    case TAU_METADATA_TYPE_ARRAY:
      Tau_XML_writeAttribute(out, metadata->data.aval, newline);
	  break;
    case TAU_METADATA_TYPE_OBJECT:
	  for (int i = 0 ; i < metadata->data.oval->count; i++) {
	    Tau_metadata_key *key = new Tau_metadata_key();
	    key->name = strdup(metadata->data.oval->names[i]);
        Tau_XML_writeAttribute(out, key, metadata->data.oval->values[i], newline);
      }
	  break;
  }
  Tau_util_output (out, "</value>%s</attribute>%s", endl, endl);
}

/*********************************************************************
 * writes an attribute entity with an int value
 ********************************************************************/
void Tau_XML_writeAttribute(Tau_util_outputDevice *out, const char *name, const int value, bool newline) {
  char str[4096];
  snprintf (str, sizeof(str),  "%d", value);
  Tau_XML_writeAttribute(out, name, str, newline);
}

/*********************************************************************
 * writes an XML time attribute
 ********************************************************************/
int Tau_XML_writeTime(Tau_util_outputDevice *out, bool newline) {
   time_t theTime = time(NULL);

   const char *endl = "";
   if (newline) {
     endl = "\n";
   }

   char buf[4096];
   struct tm *thisTime = gmtime(&theTime);
   strftime (buf,4096,"%Y-%m-%dT%H:%M:%SZ", thisTime);
   Tau_util_output (out, "<attribute><name>UTC Time</name><value>%s</value></attribute>%s", buf, endl);

   thisTime = localtime(&theTime);
   strftime (buf,4096,"%Y-%m-%dT%H:%M:%S", thisTime);

   char tzone[7];
   strftime (tzone, 7, "%z", thisTime);
   if (strlen(tzone) == 5) {
     tzone[6] = 0;
     tzone[5] = tzone[4];
     tzone[4] = tzone[3];
     tzone[3] = ':';
   }
   Tau_util_output (out, "<attribute><name>Local Time</name><value>%s%s</value></attribute>%s", buf, tzone, endl);

   // write out the timestamp (number of microseconds since epoch (unsigned long long)
#ifdef TAU_WINDOWS
   Tau_util_output (out, "<attribute><name>Timestamp</name><value>%I64d</value></attribute>%s", TauMetrics_getInitialTimeStamp(), endl);
#else
   Tau_util_output (out, "<attribute><name>Timestamp</name><value>%lld</value></attribute>%s", TauMetrics_getInitialTimeStamp(), endl);
#endif

   return 0;
}


