CC=gcc
CFLAGS=-fPIC -I.
LIBS = -L. -lsecret
LDFLAGS = -shared

app: app.o libsecret.so
	$(CC) -o app app.o $(LIBS)
libsecret.so: secret.o
	$(CC) $(LDFLAGS) -o $@ $<
secret.o: secret.c
	$(CC) $(CFLAGS) -c $<
app.o: app.c
	$(CC) $(CFLAGS) -c $<

clean:
	/bin/rm -f app.o app libsecret.so secret.o
