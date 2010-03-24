#include <TauUtil.h>
#include <stdarg.h>
#include <string.h>

int Tau_util_output(Tau_util_outputDevice *out, const char *format, ...) {

  int rs;
  va_list args;
  if (out->type == TAU_UTIL_OUTPUT_BUFFER) {
    va_start(args, format);
    rs = vsprintf(out->buffer+out->bufidx, format, args);
    va_end(args);
    out->bufidx+=rs;
    if (out->bufidx+TAU_UTIL_THRESHOLD > out->buflen) {
      out->buflen = out->buflen * 2;
      out->buffer = (char*) realloc (out->buffer, out->buflen);
    }

  } else {
    va_start(args, format);
    rs = vfprintf(out->fp, format, args);
    va_end(args);
  }
  return rs;
}

int Tau_util_readFullLine(char *line, FILE *fp) {
  int ch;
  int i = 0; 
  while ( (ch = fgetc(fp)) && ch != EOF && ch != (int) '\n') {
    line[i++] = (unsigned char) ch;
  }
  line[i] = '\0'; 
  return i; 
}

char *Tau_util_removeRuns(char *str) {
  int i, idx;
  int len; 
  // replaces runs of spaces with a single space

  if (!str) {
    return str; /* do nothing with a null string */
  }

  // also removes leading whitespace
  while (*str && *str == ' ') str++;

  len = strlen(str);
  for (i=0; i<len; i++) {
    if (str[i] == ' ') {
      idx = i+1;
      while (idx < len && str[idx] == ' ') {
	idx++;
      }
      int skip = idx - i - 1;
      for (int j=i+1; j<=len-skip; j++) {
	str[j] = str[j+skip];
      }
    }
  }
  return str;
}
