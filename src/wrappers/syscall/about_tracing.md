# About TAU_TRACE

See comments in ptrace_syscalls.c about the events.edf file to understand the issue when using TAU_TRACE=1.

## Example of exporting to chrome JSON event format

```bash
export TAU_TRACE=1
tau_exec -syscall ./hello_world
```

Now, we have a tautrace directory containing the tautrace files.

In ./tautrace, we have the files for the functions, but not the files for the syscalls.

In ./tautrace/syscall, we have the files for the syscalls only. 

```bash
ls tautrace
> events.0.edf  syscall  tautrace.0.0.0.trc
ls tautrace/syscall/
> events.0.edf  tautrace.0.0.0.trc  tautrace.0.0.1.trc
```

The file `tautrace/syscall/tautrace.0.0.0.trc` is useless since a `tautrace.0.0.0.trc` file already exists in `tautrace/`.
Other useless files like this may be created depending on the options used with tau_exec (e.g. -cupti -um will also create a useless `tautrace/syscall/tautrace.0.0.1.trc`). 
Due to the way the files are dumped, only the first files can have the same name.

```bash
rm tautrace/syscall/tautrace.0.0.0.trc
```

Now we can use `tau_merge` to merge those files. We need to associate the correct events.edf file to to tautrace files:

```bash
tau_merge -m tau.edf -e tautrace/events.0.edf tautrace/syscall/events.0.edf tautrace/tautrace.0.0.0.trc tautrace/tautrace.0.0.1.trc tau.trc
```

Then, we can convert to a json file if we want to:
```bash
tau_trace2json tau.trc tau.edf -chrome -ignoreatomic -o tau.json
```


### Example of merging with multiples files:
```bash
# After cleaning up the useless files in tautrace/syscall/
ls tautrace
> events.0.edf  syscall  tautrace.0.0.0.trc tautrace.0.0.2.trc tautrace.0.0.4.trc tautrace.0.0.6.trc tautrace.0.0.7.trc
ls tautrace/syscall/
> events.0.edf  tautrace.0.0.1.trc tautrace.0.0.3.trc tautrace.0.0.5.trc
```

For tau_merge, we can first put as many tautrace/events.0.edf as they are tautrace.trc files in tautrace/ and then as many tautrace/syscall/events.0.edf as they are tautrace.trc files in tautrace/syscall
```bash
tau_merge -m tau.edf -e tautrace/events.0.edf tautrace/events.0.edf tautrace/events.0.edf tautrace/events.0.edf tautrace/events.0.edf tautrace/syscall/events.0.edf tautrace/syscall/events.0.edf tautrace/syscall/events.0.edf tautrace.0.0.0.trc tautrace.0.0.2.trc tautrace.0.0.4.trc tautrace.0.0.6.trc tautrace.0.0.7.trc tautrace.0.0.1.trc tautrace.0.0.3.trc tautrace.0.0.5.trc tau.trc
```

A more automated way of doing that could look like this:
```bash
tau_exec -syscall [my_options] ./my_program
# Delete useless files
tautrace_list=$(ls tautrace/syscall/tautrace.* | rev | cut -d'/' -f 1 | rev | tr '\n' ' ')
for file in $tautrace_list; do
  if test -f tautrace/$file; then
    rm "tautrace/syscall/$file"
  fi
done

# Merge
cmd="tau_merge -m tau.edf -e "
number_e=$(ls tautrace/tautrace.0.* | wc -w)
cmd+=$(seq $number_e | awk '{printf "tautrace/events.0.edf "}')
number_e=$(ls tautrace/syscall/tautrace.0.* | wc -w)
cmd+=" "
cmd+=$(seq $number_e | awk '{printf "tautrace/syscall/events.0.edf "}')
cmd+=$(ls tautrace/tautrace.0.*)
cmd+=" "
cmd+=$(ls tautrace/syscall/tautrace.0.*)
cmd+=" tau.trc"
$cmd
```