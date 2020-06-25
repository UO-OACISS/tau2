.header on

select * from trial;

.header off
select "Threads:", count(*) from thread;
select "Metadata:", count(*) from metadata;
select "Metrics:", count(*) from metric;
select "Timers:", count(*) from timer;
select "Timer Values:", count(*) from timer_value;
select "Counters:", count(*) from counter;
select "Counter Values:", count(*) from counter_value;
