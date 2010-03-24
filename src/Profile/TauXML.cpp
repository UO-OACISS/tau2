#include <TauUtil.h>
#include <TauMetrics.h>

#include <string.h>
#include <stdio.h>
#include <time.h>

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
  char *str = (char *) malloc (6*strlen(s)+10);
  char *d = str;
  while (*s) {
    if ((*s == '<') || (*s == '>') || (*s == '&') || (*s == '\n')) {
      // escape these characters
      if (*s == '<') {
	strcpy (d,"&lt;");
	d+=4;
      }
      
      if (*s == '>') {
	strcpy (d,"&gt;");
	d+=4;
      }

      if (*s == '\n') {
	strcpy (d,"&#xa;");
	d+=5;
      }
      
      if (*s == '&') {
	strcpy (d,"&amp;");
	d+=5;
      }
    } else {
      *d = *s;
      d++; 
    }
    
    s++;
  }
  *d = 0;
  
  Tau_util_output (out,"%s",str);
  free (str);
}

void Tau_XML_writeTag(Tau_util_outputDevice *out, const char *tag, const char *s, bool newline) {
  Tau_util_output (out, "<%s>", tag);
  Tau_XML_writeString(out, s);
  Tau_util_output (out, "</%s>",tag);
  if (newline) {
    Tau_util_output (out, "\n");
  }
}


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


void Tau_XML_writeAttribute(Tau_util_outputDevice *out, const char *name, const int value, bool newline) {
  char str[4096];
  sprintf (str, "%d", value);
  Tau_XML_writeAttribute(out, name, str, newline);
}

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


